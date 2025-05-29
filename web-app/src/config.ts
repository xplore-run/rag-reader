interface RangeConfig {
  min: number;
  max: number;
  step: number;
  default: number;
}

interface LoanConfig {
  principal: RangeConfig;
  interestRate: RangeConfig;
  tenure: RangeConfig;
  prepayment?: RangeConfig;
}

export const config = {
  home: {
    principal: {
      min: 500000, // 5L
      max: 100000000, // 10Cr
      step: 100000, // 1L
      default: 2000000, // 20L
    },
    interestRate: {
      min: 6,
      max: 16,
      step: 0.1,
      default: 8.5,
    },
    tenure: {
      min: 5,
      max: 30,
      step: 1,
      default: 20,
    },
  } as LoanConfig,

  car: {
    principal: {
      min: 100000, // 1L
      max: 10000000, // 1Cr
      step: 50000, // 50K
      default: 1000000, // 10L
    },
    interestRate: {
      min: 6,
      max: 15,
      step: 0.1,
      default: 7.5,
    },
    tenure: {
      min: 1,
      max: 8,
      step: 1,
      default: 7,
    },
  } as LoanConfig,

  personal: {
    principal: {
      min: 50000, // 50K
      max: 5000000, // 50L
      step: 10000, // 10K
      default: 500000, // 5L
    },
    interestRate: {
      min: 10,
      max: 24,
      step: 0.1,
      default: 12,
    },
    tenure: {
      min: 1,
      max: 5,
      step: 1,
      default: 3,
    },

  } as LoanConfig,

  prepayment: {
    principal: {
      min: 100000,
      max: 10000000,
      step: 100000,
      default: 3000000,
    },
    interestRate: {
      min: 5,
      max: 20,
      step: 0.1,
      default: 8.5,
    },
    tenure: {
      min: 1,
      max: 30,
      step: 1,
      default: 20,
    },
    prepayment: {
      min: 10000,
      max: 10000000,
      step: 10000,
      default: 500000,
    },
  } as LoanConfig,

  formatting: {
    currency: {
      locale: "en-IN",
      currency: "INR",
      maximumFractionDigits: 0,
    },
  },
  formatters: {
    currency: (value: number) =>
      new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency: "INR",
        maximumFractionDigits: 0,
      }).format(value),

    percentage: (value: number) => `${value}%`,

    years: (value: number) => `${value} ${value === 1 ? "year" : "years"}`,
  },
};

export type { LoanConfig, RangeConfig };
