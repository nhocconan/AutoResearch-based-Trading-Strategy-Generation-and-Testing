# Strategy: 4h_12hEMA50_4hDonchian_Volume_v5

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.094 | +16.0% | -12.3% | 151 | FAIL |
| ETHUSDT | 0.587 | +55.9% | -11.2% | 132 | PASS |
| SOLUSDT | 0.478 | +64.1% | -28.6% | 125 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.151 | +7.7% | -8.2% | 47 | PASS |
| SOLUSDT | -0.580 | -4.3% | -15.2% | 45 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h EMA(50) as trend filter, 4-hour Donchian(20) breakout, and volume confirmation.
# Long when 12h EMA > price (bullish trend), price breaks above 4h Donchian upper band, volume > 2x average.
# Short when 12h EMA < price (bearish trend), price breaks below 4h Donchian lower band, volume > 2x average.
# Exit on trend reversal, Donchian break in opposite direction, or max 25 bars held.
# Uses position size 0.25 to balance return and drawdown. Target: 100-200 total trades over 4 years (25-50/year).
# Designed to capture trends in both bull and bear markets by using 12h trend filter, with volume to confirm breakout strength.

name = "4h_12hEMA50_4hDonchian_Volume_v5"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 12-hour EMA(50)
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4-hour Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h EMA bullish (price > EMA), price breaks above 4h Donchian upper band, volume spike
            if (close[i] > ema_12h_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: 12h EMA bearish (price < EMA), price breaks below 4h Donchian lower band, volume spike
            elif (close[i] < ema_12h_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend reversal, price breaks below Donchian lower band, or max 25 bars held
            if (close[i] < ema_12h_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal, price breaks above Donchian upper band, or max 25 bars held
            if (close[i] > ema_12h_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 18:32
