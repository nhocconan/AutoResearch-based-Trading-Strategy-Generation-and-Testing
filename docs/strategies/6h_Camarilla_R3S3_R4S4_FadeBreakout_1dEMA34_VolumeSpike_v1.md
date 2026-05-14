# Strategy: 6h_Camarilla_R3S3_R4S4_FadeBreakout_1dEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.076 | +23.3% | -13.2% | 98 | PASS |
| ETHUSDT | 0.555 | +57.4% | -14.3% | 93 | PASS |
| SOLUSDT | 1.034 | +174.5% | -22.0% | 82 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.338 | -9.5% | -13.0% | 39 | FAIL |
| ETHUSDT | 0.019 | +5.4% | -7.3% | 34 | PASS |
| SOLUSDT | 0.008 | +5.0% | -13.5% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev * 2) / 4
    range_ = high_prev - low_prev
    
    # Resistance and Support levels (focus on R3/S3 for fading, R4/S4 for breakout)
    r3 = pivot + range_ * 1.25
    s3 = pivot - range_ * 1.25
    r4 = pivot + range_ * 1.5
    s4 = pivot - range_ * 1.5
    
    # Align levels to 6h timeframe
    ema34_aligned = ema34_1d_aligned
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, pivots, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Fade at R3/S3: price touches level and reverses
            # Long: touch S3, close above it, in uptrend, volume spike
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i] and 
                close[i] > ema_trend and vol_spike_val):
                signals[i] = size
                position = 1
            # Short: touch R3, close below it, in downtrend, volume spike
            elif (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i] and 
                  close[i] < ema_trend and vol_spike_val):
                signals[i] = -size
                position = -1
            # Breakout continuation at R4/S4: strong break of extreme levels
            # Long: break above R4 with volume spike and uptrend
            elif (high[i] > r4_aligned[i] and close[i] > r4_aligned[i] and 
                  close[i] > ema_trend and vol_spike_val):
                signals[i] = size
                position = 1
            # Short: break below S4 with volume spike and downtrend
            elif (low[i] < s4_aligned[i] and close[i] < s4_aligned[i] and 
                  close[i] < ema_trend and vol_spike_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S3 (mean reversion) or trend reverses
            if low[i] <= s3_aligned[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches R3 (mean reversion) or trend reverses
            if high[i] >= r3_aligned[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_FadeBreakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 05:56
