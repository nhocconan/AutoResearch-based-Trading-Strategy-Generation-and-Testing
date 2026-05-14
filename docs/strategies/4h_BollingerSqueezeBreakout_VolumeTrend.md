# Strategy: 4h_BollingerSqueezeBreakout_VolumeTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.532 | +34.8% | -6.9% | 268 | PASS |
| ETHUSDT | 0.108 | +24.3% | -4.3% | 255 | PASS |
| SOLUSDT | 0.149 | +27.0% | -16.1% | 261 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.192 | +0.5% | -3.2% | 97 | FAIL |
| ETHUSDT | 1.063 | +16.4% | -3.1% | 86 | PASS |
| SOLUSDT | -0.044 | +5.6% | -5.6% | 83 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Bollinger Bands squeeze breakout with volume confirmation and trend filter.
- Enter long when price breaks above upper BB(20,2) + volume > 1.5x 20-period volume MA + price above 200 EMA
- Enter short when price breaks below lower BB(20,2) + volume > 1.5x 20-period volume MA + price below 200 EMA
- Exit when price crosses back inside Bollinger Bands
- Fixed position size 0.25 to manage drawdown
- Uses volatility contraction/expansion principle: squeeze precedes breakout
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
- Bollinger Bands capture volatility regimes, effective in both accumulation and distribution phases
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
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Trend filter: 200 EMA
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for 200 EMA
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20.iloc[i]) or np.isnan(std_20.iloc[i]) or 
            np.isnan(volume_ma_20.iloc[i]) or np.isnan(ema_200.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = upper_bb.iloc[i]
        lower = lower_bb.iloc[i]
        ema_val = ema_200.iloc[i]
        
        if position == 0:
            # Look for Bollinger Band breakouts with volume confirmation and trend filter
            # Long: price breaks above upper BB + volume spike + price above 200 EMA
            if price > upper and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB + volume spike + price below 200 EMA
            elif price < lower and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses back inside Bollinger Bands (mean reversion)
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back inside Bollinger Bands (mean reversion)
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerSqueezeBreakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 21:50
