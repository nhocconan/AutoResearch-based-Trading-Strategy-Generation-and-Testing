# Strategy: 4h_Donchian20_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.115 | +25.3% | -10.2% | 179 | PASS |
| ETHUSDT | 0.024 | +20.1% | -12.3% | 174 | PASS |
| SOLUSDT | 0.582 | +77.5% | -23.8% | 179 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.249 | -4.4% | -7.2% | 66 | FAIL |
| ETHUSDT | 0.535 | +13.8% | -6.8% | 57 | PASS |
| SOLUSDT | 0.430 | +12.3% | -8.4% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike (1.5x)
# Long when price breaks above 4h Donchian upper(20) AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian lower(20) AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period average
# Exit when price crosses 4h Donchian midpoint (middle of channel) OR EMA34 filter reverses
# Uses Donchian channels for clear breakout levels, 1d EMA34 for regime filter (more responsive than EMA50)
# Volume spike confirms institutional participation (moderate threshold to balance signal quality and trade frequency)
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 4h (primary timeframe as required)
# Target: 75-150 total trades over 4 years (19-37/year) to minimize fee drag

name = "4h_Donchian20_1dEMA34_VolumeSpike"
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
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20) channels
    # Upper = max(high_4h, 20), Lower = min(low_4h, 20)
    # Using rolling window with min_periods
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 4h (threshold: 1.5x for balanced filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR price < EMA34 (trend weakening)
            if close[i] < donchian_middle_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR price > EMA34 (trend weakening)
            if close[i] > donchian_middle_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 14:29
