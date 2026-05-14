# Strategy: 4h_Keltner_Channel_Breakout_With_Volume_and_12hEMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.052 | +15.5% | -15.4% | 91 | FAIL |
| ETHUSDT | 0.085 | +22.7% | -14.8% | 87 | PASS |
| SOLUSDT | 0.731 | +116.7% | -30.7% | 71 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.222 | +9.1% | -8.4% | 33 | PASS |
| SOLUSDT | -0.801 | -10.1% | -23.9% | 27 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_With_Volume_and_12hEMA34
Hypothesis: Buy when price breaks above upper Keltner channel with volume spike and above 12h EMA34; short when breaks below lower Keltner channel with volume spike and below 12h EMA34. Keltner channels use ATR for volatility, adapting to market conditions. Volume confirms institutional participation, and 12h EMA34 ensures alignment with medium-term trend. Designed for low trade frequency to minimize fee drag while capturing high-probability breakouts in both bull and bear markets.
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
    
    # Keltner Channel parameters
    atr_period = 10
    kc_mult = 2.0
    
    # Calculate ATR
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # EMA middle line
    ema_period = 20
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Keltner channels
    upper = ema + (kc_mult * atr)
    lower = ema - (kc_mult * atr)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(40, atr_period, ema_period)  # Need indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_val = upper[i]
        lower_val = lower[i]
        vol_spike = volume_spike[i]
        ema_12h_val = ema_12h_aligned[i]
        
        if position == 0:
            # Long: price > upper Keltner with volume spike and above 12h EMA34
            if price > upper_val and vol_spike and price > ema_12h_val:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Keltner with volume spike and below 12h EMA34
            elif price < lower_val and vol_spike and price < ema_12h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < lower or below 12h EMA34
            if price < lower_val or price < ema_12h_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > upper or above 12h EMA34
            if price > upper_val or price > ema_12h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Keltner_Channel_Breakout_With_Volume_and_12hEMA34"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 02:40
