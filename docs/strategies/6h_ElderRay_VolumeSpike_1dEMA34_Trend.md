# Strategy: 6h_ElderRay_VolumeSpike_1dEMA34_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.326 | +37.2% | -9.7% | 175 | PASS |
| ETHUSDT | 0.087 | +23.5% | -17.9% | 162 | PASS |
| SOLUSDT | 1.275 | +237.1% | -21.8% | 131 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.803 | -3.1% | -7.3% | 74 | FAIL |
| ETHUSDT | 0.314 | +10.8% | -8.2% | 57 | PASS |
| SOLUSDT | 0.031 | +5.4% | -12.8% | 53 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h Elder Ray + 1d EMA34 Trend + Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures
buying/selling pressure relative to trend. In strong uptrends (price > 1d EMA34),
Bull Power > 0 indicates buying momentum for longs. In strong downtrends (price < 1d EMA34),
Bear Power < 0 indicates selling pressure for shorts. Volume spike confirms institutional
participation. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
6h timeframe targets 12-37 trades/year (50-150 over 4 years).
"""

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (using close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 13, 34)  # volume MA, EMA13, 1d EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0 AND uptrend AND volume spike
            long_entry = (curr_bull_power > 0) and uptrend and vol_spike
            # Short: Bear Power < 0 AND downtrend AND volume spike
            short_entry = (curr_bear_power < 0) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Bull Power turns negative (loss of buying pressure) OR loss of uptrend
            if (curr_bull_power <= 0) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power turns positive (loss of selling pressure) OR loss of downtrend
            if (curr_bear_power >= 0) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_VolumeSpike_1dEMA34_Trend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 06:34
