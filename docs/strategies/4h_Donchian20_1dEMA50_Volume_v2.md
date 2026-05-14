# Strategy: 4h_Donchian20_1dEMA50_Volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.296 | +32.8% | -11.1% | 127 | KEEP |
| ETHUSDT | 0.218 | +30.8% | -13.9% | 121 | KEEP |
| SOLUSDT | 0.758 | +95.6% | -18.4% | 114 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.684 | +0.4% | -4.7% | 49 | DISCARD |
| ETHUSDT | 0.524 | +13.3% | -6.3% | 45 | KEEP |
| SOLUSDT | 0.184 | +8.2% | -8.2% | 37 | KEEP |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper AND close > 1d EMA50 AND volume > 2.0x average
# Short when price breaks below Donchian lower AND close < 1d EMA50 AND volume > 2.0x average
# Exit when price crosses Donchian middle (mean reversion) OR trend reversal (price crosses 1d EMA50)
# Uses 4h timeframe for optimal trade frequency, Donchian for structure, 1d EMA for trend filter, volume spike for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies.

name = "4h_Donchian20_1dEMA50_Volume_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h data (using previous 20 bars)
    if len(high_4h) >= 20:
        upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
        lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
        middle_4h = (upper_4h + lower_4h) / 2
    else:
        upper_4h = np.full_like(high_4h, np.nan)
        lower_4h = np.full_like(low_4h, np.nan)
        middle_4h = np.full_like(high_4h, np.nan)
    
    # Align Donchian levels to 4h timeframe (already aligned since calculated on 4h)
    # But we need to ensure proper alignment with look-ahead prevention
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 4h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for Donchian and EMA
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper AND close > 1d EMA50 AND volume spike
            if close[i] > upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower AND close < 1d EMA50 AND volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < middle (mean reversion) OR trend reversal (close < 1d EMA50)
            if close[i] < middle_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > middle (mean reversion) OR trend reversal (close > 1d EMA50)
            if close[i] > middle_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 14:26
