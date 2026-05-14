# Strategy: 4h_Bollinger_Bands_Breakout_With_Volume_and_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.285 | +33.6% | -12.7% | 172 | PASS |
| ETHUSDT | 0.216 | +31.4% | -16.6% | 167 | PASS |
| SOLUSDT | 0.726 | +94.9% | -22.9% | 133 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.663 | -8.4% | -9.7% | 66 | FAIL |
| ETHUSDT | 0.673 | +15.9% | -7.1% | 54 | PASS |
| SOLUSDT | 0.172 | +8.1% | -12.1% | 45 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Bollinger_Bands_Breakout_With_Volume_and_Trend_v1
Hypothesis: Buy when price breaks above upper Bollinger Band (20,2) with volume spike and above 12h EMA34 trend; sell when price breaks below lower band with volume spike and below 12h EMA34. Bollinger Bands capture volatility expansion, volume confirms institutional interest, and 12h EMA34 ensures alignment with medium-term trend. Designed for low trade frequency (<30/year) to minimize fee drift while capturing explosive moves in both bull and bear markets.
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
    close_series = pd.Series(close)
    ma = close_series.rolling(window=20, min_periods=20).mean()
    std = close_series.rolling(window=20, min_periods=20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    upper_band = upper.values
    lower_band = lower.values
    
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
    
    start_idx = 40  # Need BB and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_12h_val = ema_12h_aligned[i]
        vol_spike = volume_spike[i]
        upper = upper_band[i]
        lower = lower_band[i]
        
        if position == 0:
            # Long: price > upper BB with volume spike and above 12h EMA34
            if price > upper and vol_spike and price > ema_12h_val:
                signals[i] = 0.25
                position = 1
            # Short: price < lower BB with volume spike and below 12h EMA34
            elif price < lower and vol_spike and price < ema_12h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < middle band (mean reversion) or below 12h EMA34
            if price < ma.values[i] or price < ema_12h_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > middle band or above 12h EMA34
            if price > ma.values[i] or price > ema_12h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Bollinger_Bands_Breakout_With_Volume_and_Trend_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 02:24
