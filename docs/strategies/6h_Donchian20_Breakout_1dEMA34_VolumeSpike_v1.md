# Strategy: 6h_Donchian20_Breakout_1dEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.126 | +12.3% | -18.1% | 74 | FAIL |
| ETHUSDT | 0.158 | +28.3% | -14.5% | 78 | PASS |
| SOLUSDT | 0.929 | +158.6% | -20.5% | 65 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.509 | +15.2% | -8.8% | 24 | PASS |
| SOLUSDT | -0.300 | -1.6% | -18.5% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses Donchian channel from previous 20 periods for structure (proven edge on SOL)
# Only trade breakouts above upper channel or below lower channel in direction of 1d EMA34 trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# 1d EMA34 provides smoother trend than shorter EMAs, reducing whipsaw in ranging markets
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by following the 1d EMA34 trend direction.

name = "6h_Donchian20_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian calculation
    
    for i in range(start_idx, n):
        # Donchian channel from previous 20 periods (lookback)
        lookback_start = max(0, i-20)
        lookback_end = i  # exclusive, so we use periods [i-20, i-1]
        if lookback_end - lookback_start < 20:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[lookback_start:lookback_end])
        lowest_low = np.min(low[lookback_start:lookback_end])
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above upper Donchian AND above 1d EMA34 (uptrend)
                if curr_close > highest_high and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below lower Donchian AND below 1d EMA34 (downtrend)
                elif curr_close < lowest_low and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below lower Donchian or below 1d EMA34
            if curr_close < lowest_low or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above upper Donchian or above 1d EMA34
            if curr_close > highest_high or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 09:07
