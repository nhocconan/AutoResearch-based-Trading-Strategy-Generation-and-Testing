# Strategy: 4h_Supertrend_1dEMA34_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.287 | +36.9% | -10.7% | 201 | KEEP |
| ETHUSDT | 0.345 | +44.8% | -14.1% | 196 | KEEP |
| SOLUSDT | 0.817 | +146.8% | -32.5% | 195 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.527 | -0.9% | -8.2% | 69 | DISCARD |
| ETHUSDT | 0.436 | +13.7% | -10.0% | 67 | KEEP |
| SOLUSDT | 0.683 | +21.0% | -9.4% | 61 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend (ATR=10, mult=3.0) with 1-day EMA34 trend filter and volume confirmation.
# Supertrend identifies trend direction using ATR-based bands. 
# Long when price closes above Supertrend upper band, short when closes below lower band.
# Uses 1-day EMA34 for trend filter to align with higher timeframe direction.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (captures sustained uptrends) and bear markets (captures sustained downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR for Supertrend (period=10)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Initialize Supertrend arrays
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[9] = upper_band[9]  # Start with upper band
    direction[9] = 1
    
    for i in range(10, n):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        if close[i] <= supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = 1
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above Supertrend (uptrend), uptrend on 1d EMA34, volume
        if (close[i] > supertrend[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price below Supertrend (downtrend), downtrend on 1d EMA34, volume
        elif (close[i] < supertrend[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and close[i] <= supertrend[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= supertrend[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Supertrend_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 19:56
