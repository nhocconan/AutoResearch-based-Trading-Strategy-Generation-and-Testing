# Strategy: 4h_BB_Breakout_Volume_ATR

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.388 | +41.1% | -12.3% | 225 | PASS |
| ETHUSDT | 0.114 | +25.3% | -14.2% | 231 | PASS |
| SOLUSDT | 0.648 | +91.6% | -23.3% | 201 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.401 | -7.9% | -10.8% | 81 | FAIL |
| ETHUSDT | 0.276 | +9.9% | -9.8% | 76 | PASS |
| SOLUSDT | -0.396 | -2.3% | -14.8% | 70 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with volume confirmation and 1d ATR-based volatility filter.
# In bull markets, price breaks above upper BB; in bear markets, breaks below lower BB.
# Volume confirms conviction; 1d ATR filter avoids trading in excessively volatile or quiet conditions.
# Uses Bollinger Bands for volatility breakout, volume for confirmation, ATR for regime filter.
name = "4h_BB_Breakout_Volume_ATR"
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
    
    # Bollinger Bands (20, 2) on 4h close
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
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
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(vol_ema20[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price > upper BB + volume confirmation + volatility filter
            if (price > upper_band[i] and vol_confirm[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < lower BB + volume confirmation + volatility filter
            elif (price < lower_band[i] and vol_confirm[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below middle BB
            if price < sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above middle BB
            if price > sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 01:42
