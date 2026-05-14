# Strategy: 4h_Donchian20_Breakout_VolumeSpike_12hEMA50_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.112 | +25.1% | -17.9% | 97 | KEEP |
| ETHUSDT | 0.071 | +21.0% | -17.4% | 93 | KEEP |
| SOLUSDT | 1.192 | +287.7% | -27.6% | 93 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.118 | -7.6% | -12.5% | 39 | DISCARD |
| ETHUSDT | 0.848 | +24.2% | -7.4% | 29 | KEEP |
| SOLUSDT | 0.181 | +8.4% | -14.1% | 32 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-bar volume MA) and 12h EMA50 trend filter
# Donchian channels provide objective breakout levels; volume confirms institutional participation.
# 12h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades.
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull (breakouts with volume) and bear (failed reversals reverse quickly at strong levels).

name = "4h_Donchian20_Breakout_VolumeSpike_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) on 12h close
    ema_12h_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Donchian(20) channels on 4h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for 12h EMA and 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > highest_20[i-1]  # Break above upper channel
        breakout_down = curr_close < lowest_20[i-1]  # Break below lower channel
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, price above 12h EMA50, volume spike
            if breakout_up and curr_close > ema_12h_50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, price below 12h EMA50, volume spike
            elif breakout_down and curr_close < ema_12h_50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or price below 12h EMA50
            if curr_close < lowest_20[i] or curr_close < ema_12h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or price above 12h EMA50
            if curr_close > highest_20[i] or curr_close > ema_12h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 18:02
