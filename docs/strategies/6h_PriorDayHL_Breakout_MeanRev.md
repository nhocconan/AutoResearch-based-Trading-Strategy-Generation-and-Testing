# Strategy: 6h_PriorDayHL_Breakout_MeanRev

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.539 | +44.0% | -6.8% | 129 | PASS |
| ETHUSDT | 0.223 | +30.8% | -10.9% | 116 | PASS |
| SOLUSDT | 0.561 | +69.3% | -18.6% | 110 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.740 | -0.8% | -5.4% | 49 | FAIL |
| ETHUSDT | 1.184 | +24.3% | -5.9% | 43 | PASS |
| SOLUSDT | -0.328 | +0.9% | -7.1% | 44 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: On the 6-hour timeframe, price often respects the previous day's high/low as key support/resistance levels. 
We combine this with a 1-day EMA50 trend filter and volume confirmation to capture breakouts and reversals. 
Long when price breaks above prior day's high with volume > 2x average and price above daily EMA50. 
Short when price breaks below prior day's low with volume > 2x average and price below daily EMA50. 
Exit when price returns to the prior day's midpoint (mean reversion) or on opposite breakout. 
Designed for 6h to work in trending (breakouts) and ranging (mean reversion to mid-point) markets with ~15-25 trades per year.
"""

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
    
    # Get daily data for prior day's high/low and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's high and low (use shift(1) to avoid look-ahead: use completed day's levels)
    phigh = df_1d['high'].shift(1).values
    plow = df_1d['low'].shift(1).values
    pclose = df_1d['close'].values
    
    # Prior day's midpoint for mean reversion exit
    pmid = (phigh + plow) / 2
    
    # Calculate 1-day EMA50 for trend filter (use prior day's close to avoid look-ahead)
    ema_50 = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all daily levels to 6h timeframe (waits for daily bar to close)
    phigh_6h = align_htf_to_ltf(prices, df_1d, phigh)
    plow_6h = align_htf_to_ltf(prices, df_1d, plow)
    pmid_6h = align_htf_to_ltf(prices, df_1d, pmid)
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(phigh_6h[i]) or np.isnan(plow_6h[i]) or np.isnan(pmid_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior day's high with volume spike and above daily EMA50
            if price > phigh_6h[i] and vol > 2.0 * vol_ma and price > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior day's low with volume spike and below daily EMA50
            elif price < plow_6h[i] and vol > 2.0 * vol_ma and price < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior day's midpoint (mean reversion) OR breaks below prior day's low (invalidates breakout)
            if price < pmid_6h[i] or price < plow_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to prior day's midpoint (mean reversion) OR breaks above prior day's high (invalidates breakout)
            if price > pmid_6h[i] or price > phigh_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_PriorDayHL_Breakout_MeanRev"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 22:22
