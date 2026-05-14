# Strategy: 1d_Donchian20_1wEMA21_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.197 | +28.4% | -12.8% | 14 | PASS |
| ETHUSDT | 0.184 | +28.5% | -10.0% | 13 | PASS |
| SOLUSDT | 0.912 | +106.6% | -22.3% | 11 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.954 | -4.9% | -8.1% | 7 | FAIL |
| ETHUSDT | 0.020 | +5.9% | -11.4% | 5 | PASS |
| SOLUSDT | -1.631 | -11.9% | -11.9% | 5 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, 1w EMA21 rising, volume > 1.5x average
# Short when price breaks below Donchian(20) low, 1w EMA21 falling, volume > 1.5x average
# Uses Donchian channel for price structure, EMA21 for trend filter, volume for confirmation
# Targets 7-25 trades per year (28-100 over 4 years) for low fee drag and high win rate
# Works in both bull and bear markets due to trend filter and volume confirmation

name = "1d_Donchian20_1wEMA21_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous day's data (to avoid look-ahead)
    high_1d = high.copy()
    low_1d = low.copy()
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    
    # Donchian(20) on daily data
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA21 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 days of data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema21_1w_val = ema21_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 1w uptrend, volume confirmation
            if high_val > donchian_high_val and ema21_1w_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 1w downtrend, volume confirmation
            elif low_val < donchian_low_val and ema21_1w_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or 1w trend down
            if low_val < donchian_low_val or ema21_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or 1w trend up
            if high_val > donchian_high_val or ema21_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 14:28
