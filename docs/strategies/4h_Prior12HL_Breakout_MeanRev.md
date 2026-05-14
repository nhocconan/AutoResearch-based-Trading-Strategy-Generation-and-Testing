# Strategy: 4h_Prior12HL_Breakout_MeanRev

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.178 | +27.8% | -10.9% | 229 | PASS |
| ETHUSDT | 0.001 | +19.7% | -13.7% | 220 | PASS |
| SOLUSDT | 0.552 | +65.9% | -25.8% | 184 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.019 | -10.8% | -11.8% | 93 | FAIL |
| ETHUSDT | 0.639 | +14.6% | -7.7% | 79 | PASS |
| SOLUSDT | 0.279 | +9.4% | -10.4% | 57 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: On the 4-hour timeframe, price often respects the 12-hour high/low as key support/resistance levels.
We combine this with a 12-hour EMA34 trend filter and volume confirmation to capture breakouts.
Long when price breaks above prior 12h high with volume > 2x average and price above 12h EMA34.
Short when price breaks below prior 12h low with volume > 2x average and price below 12h EMA34.
Exit when price returns to the prior 12h midpoint (mean reversion) or on opposite breakout.
Designed for 4h to work in trending (breakouts) and ranging (mean reversion to mid-point) markets with ~20-30 trades per year.
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
    
    # Get 12h data for prior period's high/low and EMA34
    df_12h = get_htf_data(prices, '12h')
    
    # Prior 12h high and low (use shift(1) to avoid look-ahead: use completed period's levels)
    phigh = df_12h['high'].shift(1).values
    plow = df_12h['low'].shift(1).values
    pclose = df_12h['close'].values
    
    # Prior 12h midpoint for mean reversion exit
    pmid = (phigh + plow) / 2
    
    # Calculate 12h EMA34 for trend filter (use prior period's close to avoid look-ahead)
    ema_34 = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 12h levels to 4h timeframe (waits for 12h bar to close)
    phigh_4h = align_htf_to_ltf(prices, df_12h, phigh)
    plow_4h = align_htf_to_ltf(prices, df_12h, plow)
    pmid_4h = align_htf_to_ltf(prices, df_12h, pmid)
    ema_34_4h = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(phigh_4h[i]) or np.isnan(plow_4h[i]) or np.isnan(pmid_4h[i]) or
            np.isnan(ema_34_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior 12h high with volume spike and above 12h EMA34
            if price > phigh_4h[i] and vol > 2.0 * vol_ma and price > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior 12h low with volume spike and below 12h EMA34
            elif price < plow_4h[i] and vol > 2.0 * vol_ma and price < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior 12h midpoint (mean reversion) OR breaks below prior 12h low (invalidates breakout)
            if price < pmid_4h[i] or price < plow_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to prior 12h midpoint (mean reversion) OR breaks above prior 12h high (invalidates breakout)
            if price > pmid_4h[i] or price > phigh_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Prior12HL_Breakout_MeanRev"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 22:23
