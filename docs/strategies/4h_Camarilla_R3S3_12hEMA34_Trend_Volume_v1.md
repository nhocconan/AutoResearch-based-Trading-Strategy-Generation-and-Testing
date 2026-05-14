# Strategy: 4h_Camarilla_R3S3_12hEMA34_Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.381 | +35.8% | -11.3% | 242 | PASS |
| ETHUSDT | 0.019 | +21.1% | -11.3% | 221 | PASS |
| SOLUSDT | 0.123 | +25.7% | -12.4% | 181 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.237 | -3.1% | -5.3% | 101 | FAIL |
| ETHUSDT | 1.545 | +26.6% | -4.9% | 82 | PASS |
| SOLUSDT | 0.482 | +11.5% | -6.8% | 62 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Uses 4h Camarilla pivot levels (R3/S3) for structure-based breakouts with institutional validation
# 12h EMA34 ensures alignment with intermediate trend to avoid counter-trend whipsaws
# Volume spike (>2.0 * 20-period EMA) confirms strong participation
# Designed for low trade frequency: ~20-40 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# Uses actual 4h Camarilla calculations (not resampled) for structure

name = "4h_Camarilla_R3S3_12hEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h data for Camarilla pivots (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (R3, S3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formula: Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 2
    # S3 = Pivot - (H - L) * 1.1 / 2
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r3_4h = pivot_4h + (high_4h - low_4h) * 1.1 / 2.0
    s3_4h = pivot_4h - (high_4h - low_4h) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (completed 4h bar only)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
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
        
        # Determine trend bias from 12h EMA34
        bullish_bias = close[i] > ema_34_aligned[i]
        bearish_bias = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above R3 with volume spike
                if close[i] > r3_aligned[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below S3 with volume spike
                if close[i] < s3_aligned[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA34
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 or price below 12h EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or price above 12h EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 00:17
