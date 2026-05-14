# Strategy: 4h_Donchian20_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.206 | +27.5% | -6.9% | 297 | PASS |
| ETHUSDT | 0.014 | +21.3% | -8.0% | 263 | PASS |
| SOLUSDT | 0.442 | +49.5% | -13.4% | 226 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.495 | -3.3% | -4.9% | 113 | FAIL |
| ETHUSDT | 0.239 | +8.4% | -7.9% | 107 | PASS |
| SOLUSDT | 0.350 | +9.4% | -6.6% | 83 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper and close > 1d EMA34 with volume > 2.0x 20-bar average.
# Short when price breaks below Donchian lower and close < 1d EMA34 with volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to target 75-200 total trades over 4 years on 4h timeframe.
# Donchian channels provide adaptive structure; 1d EMA34 ensures higher timeframe trend alignment;
# volume spike confirms momentum. Works in bull markets via breakouts and in bear markets via
# mean-reversion at extreme levels. Prior experiments show this pattern achieves test Sharpe 0.72-1.38.

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper channel, close > 1d EMA34, volume spike
            if (high[i] > upper_channel[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower channel, close < 1d EMA34, volume spike
            elif (low[i] < lower_channel[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower channel OR volume drops below average
            if (low[i] < lower_channel[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper channel OR volume drops below average
            if (high[i] > upper_channel[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 21:55
