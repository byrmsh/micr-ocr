import { Container, getContainer } from "@cloudflare/containers";

export class MicrContainer extends Container<Env> {
  // The FastAPI app inside the image listens here (see Dockerfile CMD).
  defaultPort = 8080;
  // Keep a warm instance between bursts of requests, then release it. The demo
  // page ships pre-computed sample results so first-click never waits on a cold start.
  sleepAfter = "10m";
}

export interface Env {
  MICR: DurableObjectNamespace<MicrContainer>;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Stateless service: route every request to a single warm instance.
    return getContainer(env.MICR, "singleton").fetch(request);
  },
};
