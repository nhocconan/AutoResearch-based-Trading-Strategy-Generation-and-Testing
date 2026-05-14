# Strategy: 12h_Prior1D_HL_Breakout_Volume_1DTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.251 | +32.2% | -11.1% | 95 | PASS |
| ETHUSDT | 0.044 | +21.0% | -14.9% | 91 | PASS |
| SOLUSDT | 0.713 | +103.2% | -26.6% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.013 | -4.8% | -8.2% | 37 | FAIL |
| ETHUSDT | 0.020 | +5.5% | -8.0% | 30 | PASS |
| SOLUSDT | 0.131 | +7.4% | -11.4% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h 1D High/Low Breakout with Volume and 1D Trend Filter
Long: Price breaks above prior 1D high + volume > 1.5x 12h volume MA + price > 1D EMA50
Short: Price breaks below prior 1D low + volume > 1.5x 12h volume MA + price < 1D EMA50
Exit: Opposite break of prior 1D level
Uses 1D EMA50 (not 12h) to align with longer-term bias and reduce false breakouts in chop
Target: 20-30 trades/year per symbol
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
    
    # Get 1D data for prior high/low and trend filter
    df_1d = get_htf_data(prices, '1d')
    prior_1d_high = df_1d['high'].shift(1)  # Prior day's high
    prior_1d_low = df_1d['low'].shift(1)    # Prior day's low
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    prior_1d_high_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_high.values)
    prior_1d_low_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_low.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume moving average (24-period for confirmation)
    df_12h = get_htf_data(prices, '12h')
    volume_ma_24 = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean()
    volume_ma_24_12h = align_htf_to_ltf(prices, df_12h, volume_ma_24.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1d_high_aligned[i]) or np.isnan(prior_1d_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_12h[i]
        
        if position == 0:
            # Long: break above prior 1D high + volume + 1D trend
            if price > prior_1d_high_aligned[i] and vol > 1.5 * vol_ma and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior 1D low + volume + 1D trend
            elif price < prior_1d_low_aligned[i] and vol > 1.5 * vol_ma and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior 1D low
            if price < prior_1d_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior 1D high
            if price > prior_1d_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Prior1D_HL_Breakout_Volume_1DTrend"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-17 23:01
