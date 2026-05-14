# Strategy: 6h_Donchian20_1dEMA34_VolumeFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.443 | +52.0% | -13.0% | 160 | PASS |
| ETHUSDT | 0.384 | +51.5% | -13.9% | 207 | PASS |
| SOLUSDT | 0.687 | +128.5% | -32.4% | 258 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.237 | +1.9% | -8.3% | 71 | FAIL |
| ETHUSDT | 0.700 | +21.8% | -10.8% | 54 | PASS |
| SOLUSDT | 0.610 | +20.7% | -12.2% | 55 | PASS |

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
    
    # Get daily data for indicators (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily (upper and lower bands)
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 34-period EMA on daily for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily data to 6h timeframe (primary)
    upper_channel_6h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_6h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 6h volume spike indicator (volume > 2.0x 50-period average)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_6h[i]) or np.isnan(lower_channel_6h[i]) or 
            np.isnan(ema_34_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_34_6h[i]
        downtrend = close[i] < ema_34_6h[i]
        
        # Volume confirmation: require volume spike
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume spike
            if close[i] > upper_channel_6h[i] and uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume spike
            elif close[i] < lower_channel_6h[i] and downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses
            if (close[i] < lower_channel_6h[i]) or (not uptrend):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses
            if (close[i] > upper_channel_6h[i]) or (not downtrend):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA34_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 18:39
