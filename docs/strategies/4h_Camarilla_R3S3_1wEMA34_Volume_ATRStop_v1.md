# Strategy: 4h_Camarilla_R3S3_1wEMA34_Volume_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.232 | +29.7% | -7.9% | 135 | PASS |
| ETHUSDT | 0.127 | +25.9% | -20.0% | 124 | PASS |
| SOLUSDT | 1.007 | +110.3% | -15.0% | 95 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.974 | -1.1% | -6.8% | 44 | FAIL |
| ETHUSDT | 0.860 | +16.6% | -8.9% | 38 | PASS |
| SOLUSDT | 0.043 | +6.1% | -11.3% | 39 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot R3/S3 levels for breakout entries with volume confirmation and 1w EMA34 trend filter.
# Long when: price breaks above 1d Camarilla R3 with volume > 2.0x 20-bar average AND close > 1w EMA34 (bullish trend)
# Short when: price breaks below 1d Camarilla S3 with volume > 2.0x 20-bar average AND close < 1w EMA34 (bearish trend)
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 1w EMA34 for robust trend filtering (avoids whipsaw in chop), 1d Camarilla R3/S3 for precise breakout levels, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_Camarilla_R3S3_1wEMA34_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate typical price and pivot point for Camarilla
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    pivot_1d = typical_price_1d  # Camarilla uses typical price as pivot
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 4h timeframe (completed 1d bar only)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 4h ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for ATR and volume calculations)
    start_idx = 20 + 5
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 1d Camarilla R3 with volume spike AND bullish trend (close > 1w EMA34)
            if close[i] > r3_1d_aligned[i] and volume_spike[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below 1d Camarilla S3 with volume spike AND bearish trend (close < 1w EMA34)
            elif close[i] < s3_1d_aligned[i] and volume_spike[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 02:30
