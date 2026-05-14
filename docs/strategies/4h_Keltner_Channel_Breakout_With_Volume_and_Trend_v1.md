# Strategy: 4h_Keltner_Channel_Breakout_With_Volume_and_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.151 | +26.9% | -15.5% | 146 | PASS |
| ETHUSDT | 0.338 | +38.7% | -10.7% | 133 | PASS |
| SOLUSDT | 0.449 | +58.4% | -24.6% | 120 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.390 | -6.4% | -7.6% | 59 | FAIL |
| ETHUSDT | 0.872 | +19.2% | -7.2% | 46 | PASS |
| SOLUSDT | -0.062 | +4.2% | -13.3% | 44 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_With_Volume_and_Trend_v1
Hypothesis: Buy when price breaks above upper Keltner Channel (20, 2*ATR) with volume spike and above 12h EMA34 trend; sell when price breaks below lower Keltner Channel with volume spike and below 12h EMA34. Keltner Channels adapt better than Bollinger Bands to trending markets, capturing volatility expansion with dynamic bands. Volume confirms institutional interest, and 12h EMA34 ensures alignment with medium-term trend. Designed for low trade frequency (<30/year) to minimize fee drift while capturing explosive moves in both bull and bear markets.
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
    
    # Keltner Channels (20, 2*ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    close_series = pd.Series(close)
    ma = close_series.rolling(window=20, min_periods=20).mean()
    upper_keltner = ma + 2 * atr
    lower_keltner = ma - 2 * atr
    upper = upper_keltner.values
    lower = lower_keltner.values
    middle = ma.values
    
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
    
    start_idx = 40  # Need Keltner and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(upper[i]) or
            np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_12h_val = ema_12h_aligned[i]
        vol_spike = volume_spike[i]
        upper_val = upper[i]
        lower_val = lower[i]
        middle_val = middle[i]
        
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
            # Exit: price < middle band or below 12h EMA34
            if price < middle_val or price < ema_12h_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > middle band or above 12h EMA34
            if price > middle_val or price > ema_12h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Keltner_Channel_Breakout_With_Volume_and_Trend_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 02:26
