# Strategy: 4h_Donchian20_Volume_1dEMA34_Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.186 | +29.5% | -17.2% | 98 | PASS |
| ETHUSDT | 0.248 | +35.1% | -12.2% | 103 | PASS |
| SOLUSDT | 0.766 | +122.5% | -27.6% | 97 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.178 | -6.5% | -10.2% | 46 | FAIL |
| ETHUSDT | 0.229 | +9.2% | -10.1% | 36 | PASS |
| SOLUSDT | 0.347 | +11.8% | -14.4% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, price > 1d EMA34, and volume > 2.0x 20-bar average
# Short when price breaks below Donchian(20) low, price < 1d EMA34, and volume > 2.0x 20-bar average
# Uses 1d EMA for higher timeframe trend alignment (matches experiment HTF)
# Volume spike confirms breakout strength
# Discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (19-50/year on 4h) to avoid fee drag
# Works in bull (breakouts above rising EMA) and bear (breakdowns below falling EMA)

name = "4h_Donchian20_Volume_1dEMA34_Trend_Volume_v1"
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
    open_ = prices['open'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) on 4h (using previous 20 bars, not including current)
    # Highest high of previous 20 bars
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lowest low of previous 20 bars
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20) + 1  # EMA(34) + Donchian(20) + volume MA(20) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian high, price > 1d EMA34, volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low, price < 1d EMA34, volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian low or price < 1d EMA34
            if (close[i] < lowest_low[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian high or price > 1d EMA34
            if (close[i] > highest_high[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 01:04
