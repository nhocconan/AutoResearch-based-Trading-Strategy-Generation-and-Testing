# Strategy: 4h_DailyRange_Breakout_Volume_Momentum

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.424 | +37.2% | -6.0% | 280 | PASS |
| ETHUSDT | 0.374 | +37.5% | -9.6% | 243 | PASS |
| SOLUSDT | 0.479 | +55.0% | -24.2% | 222 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.689 | -6.5% | -8.4% | 113 | FAIL |
| ETHUSDT | 0.493 | +11.8% | -9.4% | 103 | PASS |
| SOLUSDT | 0.187 | +8.1% | -7.8% | 77 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Daily Range Breakout with Volume Spike and Momentum Filter
Hypothesis: The previous day's high and low act as key support/resistance levels.
Breakouts beyond these levels with volume confirmation capture momentum moves.
Works in both bull and bear markets by requiring volume confirmation to avoid false breakouts
and using a momentum filter to align with short-term price action.
Designed for 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for previous day's high/low (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high = df_d['high'].shift(1).values
    prev_low = df_d['low'].shift(1).values
    
    # Align to 4h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Momentum filter: 4-period RSI > 50 for long, < 50 for short
    # Using close prices for RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=4, min_periods=4).mean()
    avg_loss = loss.rolling(window=4, min_periods=4).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ph = prev_high_aligned[i]
        pl = prev_low_aligned[i]
        
        if position == 0:
            # Long: break above previous day's high with volume spike and bullish momentum
            if price > ph and volume_spike[i] and rsi_values[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: break below previous day's low with volume spike and bearish momentum
            elif price < pl and volume_spike[i] and rsi_values[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to previous day's low or momentum turns bearish
            if price <= pl or rsi_values[i] < 50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to previous day's high or momentum turns bullish
            if price >= ph or rsi_values[i] > 50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_DailyRange_Breakout_Volume_Momentum"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 01:25
