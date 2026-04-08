# Strategy: 4h_adx_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.242 | +32.5% | -12.3% | 220 | PASS |
| ETHUSDT | 0.538 | +56.3% | -11.9% | 207 | PASS |
| SOLUSDT | 0.398 | +53.9% | -16.4% | 169 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.358 | -7.4% | -10.2% | 76 | FAIL |
| ETHUSDT | 0.496 | +13.2% | -9.1% | 69 | PASS |
| SOLUSDT | -0.114 | +3.2% | -15.9% | 58 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_adx_trend_volume_v1
Hypothesis: On 4h timeframe, use ADX to detect strong trends (ADX > 25) and +DI/-DI for direction, with volume confirmation for institutional participation. Enter long when ADX > 25, +DI > -DI, and volume > 2x average; enter short when ADX > 25, -DI > +DI, and volume > 2x average. Exit when ADX falls below 20 (trend weakening) or opposite DI crossover. This strategy captures strong trending moves with volume confirmation, reducing false signals and trade frequency. Works in bull/bear via ADX trend filter and directional cues. Targets 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adx_trend_volume_v1"
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
    
    # Calculate ADX and DI on 4h data
    period = 14
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    tr_sum = np.where(tr_sum == 0, 1e-10, tr_sum)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # ADX calculation
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(period, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if ADX falls below 20 (trend weakening)
            if adx[i] < 20:
                exit_long = True
            # Exit if -DI crosses above +DI (trend reversal)
            elif minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if ADX falls below 20 (trend weakening)
            if adx[i] < 20:
                exit_short = True
            # Exit if +DI crosses above -DI (trend reversal)
            elif plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Strong trend (ADX > 25), +DI > -DI, and volume confirmation
            if adx[i] > 25 and plus_di[i] > minus_di[i] and vol_confirm:
                long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Strong trend (ADX > 25), -DI > +DI, and volume confirmation
            if adx[i] > 25 and minus_di[i] > plus_di[i] and vol_confirm:
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 16:18
