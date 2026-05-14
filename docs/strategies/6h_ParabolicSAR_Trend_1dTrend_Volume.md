# Strategy: 6h_ParabolicSAR_Trend_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.070 | +17.7% | -11.1% | 158 | FAIL |
| ETHUSDT | 0.022 | +20.7% | -14.6% | 142 | PASS |
| SOLUSDT | 0.623 | +75.3% | -14.7% | 130 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.532 | +31.0% | -5.3% | 50 | PASS |
| SOLUSDT | 0.132 | +7.4% | -7.8% | 45 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "6h_ParabolicSAR_Trend_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter and Parabolic SAR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Parabolic SAR on 6h data
    sar = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    af = 0.02
    max_af = 0.2
    ep = high[0]  # extreme point
    sar[0] = low[0]
    
    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend[i-1] == 1:  # uptrend
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                ep = low[i]
                af = 0.02
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = 0.02
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(sar[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above SAR (uptrend) + above daily EMA50 + volume spike
            if close[i] > sar[i] and close[i] > ema_50_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below SAR (downtrend) + below daily EMA50 + volume spike
            elif close[i] < sar[i] and close[i] < ema_50_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below SAR or below daily EMA50
            if close[i] < sar[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above SAR or above daily EMA50
            if close[i] > sar[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 03:55
