# Strategy: 6h_Camarilla_R3S3_1dEMA34_Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.335 | +37.6% | -10.5% | 107 | PASS |
| ETHUSDT | 0.068 | +22.4% | -13.5% | 94 | PASS |
| SOLUSDT | 0.983 | +165.6% | -25.2% | 76 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.503 | -10.2% | -16.2% | 43 | FAIL |
| ETHUSDT | 0.274 | +9.9% | -11.0% | 31 | PASS |
| SOLUSDT | -0.139 | +2.1% | -15.7% | 27 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d Camarilla pivot levels (R3/S3) for structure-based breakouts with institutional validation
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws
# Volume spike (>2.0 * 20-period EMA) confirms strong participation
# Designed for low trade frequency: ~15-25 trades/year per symbol with 0.28 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# Uses actual 1d Camarilla calculations (not resampled) for structure

name = "6h_Camarilla_R3S3_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 2
    # S3 = Pivot - (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (strict filter)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for EMA34 and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_aligned[i]
        bearish_bias = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above R3 with volume spike
                if close[i] > r3_aligned[i-1] and volume_spike[i]:
                    signals[i] = 0.28
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below S3 with volume spike
                if close[i] < s3_aligned[i-1] and volume_spike[i]:
                    signals[i] = -0.28
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA34
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 or price below 1d EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or price above 1d EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals
```

## Last Updated
2026-05-02 00:15
