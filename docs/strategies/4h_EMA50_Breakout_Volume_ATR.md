# Strategy: 4h_EMA50_Breakout_Volume_ATR

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.209 | +32.0% | -17.5% | 140 | PASS |
| ETHUSDT | 0.275 | +39.5% | -14.8% | 139 | PASS |
| SOLUSDT | 1.012 | +224.2% | -32.4% | 148 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.768 | -4.6% | -7.5% | 58 | FAIL |
| ETHUSDT | 0.784 | +23.0% | -8.5% | 49 | PASS |
| SOLUSDT | 0.522 | +17.5% | -10.5% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout above 12h EMA50 with volume confirmation and 1d ATR-based volatility filter.
# In bull markets, price breaks above rising EMA50; in bear markets, breaks below falling EMA50.
# Volume confirms conviction; 1d ATR filter avoids trading in excessively volatile or quiet conditions.
# Uses EMA for trend, volume for confirmation, ATR for regime filter - proven combo from DB.
name = "4h_EMA50_Breakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    # 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma50 = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d_aligned / atr_ma50
    vol_filter = (atr_ratio > 0.5) & (atr_ratio < 2.0)  # avoid extreme volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ema20[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price > 12h EMA50 + volume confirmation + volatility filter
            if (price > ema_12h_aligned[i] and vol_confirm[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < 12h EMA50 + volume confirmation + volatility filter
            elif (price < ema_12h_aligned[i] and vol_confirm[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below 12h EMA50
            if price < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above 12h EMA50
            if price > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 01:41
