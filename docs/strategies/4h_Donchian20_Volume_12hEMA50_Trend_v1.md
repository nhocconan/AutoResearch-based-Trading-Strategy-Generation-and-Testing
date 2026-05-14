# Strategy: 4h_Donchian20_Volume_12hEMA50_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.028 | +20.3% | -18.2% | 122 | PASS |
| ETHUSDT | 0.206 | +32.2% | -13.7% | 112 | PASS |
| SOLUSDT | 0.906 | +160.5% | -25.8% | 114 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.430 | -9.3% | -11.0% | 52 | FAIL |
| ETHUSDT | 0.132 | +7.4% | -9.8% | 41 | PASS |
| SOLUSDT | 0.562 | +16.8% | -12.2% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation
# Long when price breaks above Donchian upper(20), price > 12h EMA50, and volume > 1.8x 20-bar average
# Short when price breaks below Donchian lower(20), price < 12h EMA50, and volume > 1.8x 20-bar average
# Uses 12h EMA for higher timeframe trend alignment (matches experiment HTF)
# Volume spike confirms breakout strength
# Discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (19-50/year on 4h) to avoid fee drag
# Works in bull (breakouts above rising EMA) and bear (breakdowns below falling EMA)

name = "4h_Donchian20_Volume_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on 4h (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20) + 1  # EMA(50) + Donchian(20) + volume MA(20) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper, price > 12h EMA50, volume spike
            if (close[i] > high_roll[i] and 
                close[i] > ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian lower, price < 12h EMA50, volume spike
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian lower or price < 12h EMA50
            if (close[i] < low_roll[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian upper or price > 12h EMA50
            if (close[i] > high_roll[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 00:52
