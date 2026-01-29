import './Skeleton.css';

interface SkeletonProps {
  width?: string;
  height?: string;
  className?: string;
}

export function Skeleton({ width = '100%', height = '1rem', className = '' }: SkeletonProps) {
  return (
    <div
      className={`skeleton ${className}`}
      style={{ width, height }}
    />
  );
}

export function FormSkeleton() {
  return (
    <div className="form-skeleton">
      <Skeleton height="1.5rem" width="40%" className="mb-4" />
      <Skeleton height="3rem" className="mb-6" />
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="mb-4">
          <Skeleton height="1rem" width="30%" className="mb-2" />
          <Skeleton height="2.5rem" />
        </div>
      ))}
      <Skeleton height="3rem" className="mt-6" />
    </div>
  );
}
