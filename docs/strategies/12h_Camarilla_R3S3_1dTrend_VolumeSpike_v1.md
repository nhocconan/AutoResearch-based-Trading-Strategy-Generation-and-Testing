# Strategy: 12h_Camarilla_R3S3_1dTrend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.330 | +9.6% | -9.1% | 79 | FAIL |
| ETHUSDT | 0.206 | +29.6% | -7.3% | 63 | PASS |
| SOLUSDT | 0.288 | +39.7% | -26.5% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.207 | +8.5% | -6.2% | 27 | PASS |
| SOLUSDT | -1.048 | -7.9% | -15.3% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Uses Camarilla pivot levels from 1d for breakout signals, 1d EMA(34) for trend direction
# Volume spike (2.0x 20-period average) ensures participation and reduces false breakouts
# Only takes breakouts in the direction of the 1d trend to avoid counter-trend whipsaws
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend
# 1d trend filter provides strong directional bias suitable for 12h timeframe

name = "12h_Camarilla_R3S3_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    camarilla_r3 = pivot + (range_hl * 1.1 / 4.0)
    camarilla_s3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 12h volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla, EMA and volume MA)
    start_idx = 70  # max(20 for volume, 34 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 AND uptrend AND volume confirm
            if (close[i] > camarilla_r3_aligned[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND downtrend AND volume confirm
            elif (close[i] < camarilla_s3_aligned[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR trend reverses to downtrend
            if (close[i] < camarilla_s3_aligned[i] or 
                not uptrend):  # exited if price closes below 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR trend reverses to uptrend
            if (close[i] > camarilla_r3_aligned[i] or 
                not downtrend):  # exited if price closes above 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 12:05
