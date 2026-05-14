# Strategy: 6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.049 | +21.2% | -15.7% | 81 | PASS |
| ETHUSDT | 0.339 | +44.3% | -17.7% | 79 | PASS |
| SOLUSDT | 0.995 | +194.6% | -25.7% | 71 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.928 | -6.2% | -11.4% | 36 | FAIL |
| ETHUSDT | 0.653 | +19.4% | -8.4% | 25 | PASS |
| SOLUSDT | -0.339 | -3.2% | -23.0% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakouts on 6h with 12h EMA50 trend filter and volume spike confirmation. Uses discrete sizing (0.25) to limit trades (~25/year) and avoid fee drag. The 12h EMA50 provides smooth trend alignment, reducing whipsaws. Volume spike (>2.0x 20-bar avg) confirms breakout momentum. Designed for BTC/ETH robustness in bull/bear regimes via trend-following structure with strict entry conditions. R3/S3 levels offer stronger breakout signals than R1/S1 with fewer false triggers.
"""

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
    
    # Get 12h data for HTF trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from previous 12h bar (HLC of prior bar)
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h) / 4
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h) / 4
    
    # Align Camarilla levels to 6h timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Calculate 20-bar average volume for confirmation on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50, volume MA20
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R3 in uptrend with volume spike
            # Short: price breaks below Camarilla S3 in downtrend with volume spike
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema50_12h_aligned[i]) and volume_confirm
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema50_12h_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below 12h EMA50 (trend reversal)
            exit_signal = close[i] < ema50_12h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 12h EMA50 (trend reversal)
            exit_signal = close[i] > ema50_12h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 19:02
