# Strategy: 4h_donchian40_1d_supertrend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.035 | +16.1% | -18.9% | 64 | FAIL |
| ETHUSDT | 0.088 | +22.4% | -19.1% | 65 | PASS |
| SOLUSDT | 1.101 | +235.3% | -22.4% | 70 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.501 | +15.4% | -10.6% | 21 | PASS |
| SOLUSDT | -0.334 | -2.8% | -16.9% | 23 | FAIL |

## Code
```python
#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian40_1d_supertrend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Supertrend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily Supertrend (ATR=10, multiplier=3.0)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean()
    
    # Upper and Lower Bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Supertrend calculation
    supertrend = np.zeros(len(close_1d))
    direction = np.ones(len(close_1d))  # 1 = uptrend, -1 = downtrend
    
    for i in range(1, len(close_1d)):
        if close_1d.iloc[i] > upper_band.iloc[i-1]:
            direction[i] = 1
        elif close_1d.iloc[i] < lower_band.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if direction[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band.iloc[i]
        else:
            supertrend[i] = upper_band.iloc[i]
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    uptrend = close > supertrend_aligned  # Price above Supertrend = uptrend
    downtrend = close < supertrend_aligned  # Price below Supertrend = downtrend
    
    # Calculate Donchian channels (40-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=40, min_periods=40).max().values
    donchian_low = low_series.rolling(window=40, min_periods=40).min().values
    
    # Volume confirmation (40-period average)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period low
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: opposite Donchian break
        exit_long = close[i] < donchian_low[i-1]
        exit_short = close[i] > donchian_high[i-1]
        
        if position == 1:  # Long position
            # Exit on breakdown or trend reversal
            if exit_long or not uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on breakout or trend reversal
            if exit_short or not downtrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: upward breakout + uptrend + volume confirmation
            if breakout_up and uptrend[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: downward breakout + downtrend + volume confirmation
            elif breakout_down and downtrend[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 06:20
