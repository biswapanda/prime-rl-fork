{{/*
Expand the name of the chart.
*/}}
{{- define "prime-rl.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Resolve the immutable image reference generated for DGD, with the native chart
repository/tag remaining as the fallback for statefulset mode.
*/}}
{{- define "prime-rl.image" -}}
{{- if .Values.image.reference -}}
{{- .Values.image.reference -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end }}

{{/*
Resolve the logical GPU allocation for a Dynamo worker. In-process workers
declare GPUs on the Dynamo service, while OpenEngine workers deliberately keep
the main sidecar CPU-only and assign GPUs to the vllm-engine container.
*/}}
{{- define "prime-rl.dynamoWorkerGPUResources" -}}
{{- $serviceName := index . 0 -}}
{{- $service := index . 1 -}}
{{- $requestsGpu := "" -}}
{{- $limitsGpu := "" -}}
{{- $pod := default (dict) (index $service "extraPodSpec") -}}
{{- $engine := dict -}}
{{- $engineCount := 0 -}}
{{- range $container := default (list) (index $pod "containers") -}}
{{- if eq (default "" (index $container "name")) "vllm-engine" -}}
{{- $engine = $container -}}
{{- $engineCount = add1 $engineCount -}}
{{- end -}}
{{- end -}}
{{- if gt $engineCount 1 -}}
{{- fail (printf "%s must declare at most one vllm-engine" $serviceName) -}}
{{- end -}}
{{- if eq $engineCount 1 -}}
{{- if hasKey $service "resources" -}}
{{- fail (printf "%s cannot declare both service and vllm-engine GPU resources" $serviceName) -}}
{{- end -}}
{{- $resources := required (printf "%s vllm-engine resources are required" $serviceName) (index $engine "resources") -}}
{{- $requests := required (printf "%s vllm-engine resource requests are required" $serviceName) (index $resources "requests") -}}
{{- $limits := required (printf "%s vllm-engine resource limits are required" $serviceName) (index $resources "limits") -}}
{{- $requestsGpu = required (printf "%s vllm-engine GPU request is required" $serviceName) (index $requests "nvidia.com/gpu") -}}
{{- $limitsGpu = required (printf "%s vllm-engine GPU limit is required" $serviceName) (index $limits "nvidia.com/gpu") -}}
{{- else if hasKey $service "resources" -}}
{{- $resources := required (printf "%s resources are required" $serviceName) (index $service "resources") -}}
{{- $requests := required (printf "%s resource requests are required" $serviceName) (index $resources "requests") -}}
{{- $limits := required (printf "%s resource limits are required" $serviceName) (index $resources "limits") -}}
{{- $requestsGpu = required (printf "%s GPU request is required" $serviceName) (index $requests "gpu") -}}
{{- $limitsGpu = required (printf "%s GPU limit is required" $serviceName) (index $limits "gpu") -}}
{{- else -}}
{{- fail (printf "%s must declare service GPU resources or a vllm-engine container" $serviceName) -}}
{{- end -}}
{{- dict "requestsGpu" $requestsGpu "limitsGpu" $limitsGpu | toJson -}}
{{- end }}

{{/*
Reuse a supplied shared claim or derive the chart-managed claim name.
*/}}
{{- define "prime-rl.storageClaimName" -}}
{{- default (printf "%s-shared-data" .Release.Name) .Values.storage.existingClaim -}}
{{- end }}

{{- define "prime-rl.inferenceUrls" -}}
{{- if eq .Values.inference.mode "dynamoGraph" -}}
{{- printf "http://%s-frontend.%s.svc.cluster.local:8000/v1" .Release.Name .Values.namespace -}}
{{- else -}}
{{- $releaseName := .Release.Name -}}
{{- $namespace := .Values.namespace -}}
{{- $port := int .Values.inference.service.port -}}
{{- $replicas := int .Values.inference.replicas -}}
{{- $urls := list -}}
{{- range $i := until $replicas -}}
{{- $url := printf "http://%s-inference-%d.%s-inference-headless.%s.svc.cluster.local:%d/v1" $releaseName $i $releaseName $namespace $port -}}
{{- $urls = append $urls $url -}}
{{- end -}}
{{- $urls | join "," -}}
{{- end -}}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "prime-rl.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "prime-rl.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "prime-rl.labels" -}}
helm.sh/chart: {{ include "prime-rl.chart" . }}
{{ include "prime-rl.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "prime-rl.selectorLabels" -}}
app.kubernetes.io/name: {{ include "prime-rl.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Component labels
*/}}
{{- define "prime-rl.componentLabels" -}}
app: prime-rl
{{- if .Values.config.example }}
example: {{ .Values.config.example }}
{{- end }}
{{- end }}
