# Strategy: 4h_Camarilla_R4S4_1dEMA50_Volume_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.488 | +43.8% | -7.7% | 150 | PASS |
| ETHUSDT | 0.000 | +18.9% | -13.3% | 149 | PASS |
| SOLUSDT | 0.990 | +144.2% | -19.7% | 125 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.043 | +5.5% | -5.3% | 54 | FAIL |
| ETHUSDT | 1.490 | +30.4% | -6.5% | 51 | PASS |
| SOLUSDT | 0.654 | +15.9% | -10.1% | 42 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot R4/S4 levels for breakout entries with volume confirmation and 1d EMA50 trend filter.
# Long when: price breaks above 1d Camarilla R4 with volume > 2.0x 20-bar average AND close > 1d EMA50 (bullish trend)
# Short when: price breaks below 1d Camarilla S4 with volume > 2.0x 20-bar average AND close < 1d EMA50 (bearish trend)
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 1d EMA50 for robust trend filtering (avoids whipsaw in chop), 1d Camarilla R4/S4 for precise breakout levels (wider than R3/S3), volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_Camarilla_R4S4_1dEMA50_Volume_ATRStop_v1"
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
    
    # Calculate 1d Camarilla pivot levels (R4, S4)
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
    
    # Camarilla R4 and S4 levels (wider bands than R3/S3)
    r4_1d = pivot_1d + (range_1d * 1.1 / 2)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h timeframe (completed 1d bar only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 1d Camarilla R4 with volume spike AND bullish trend (close > 1d EMA50)
            if close[i] > r4_1d_aligned[i] and volume_spike[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below 1d Camarilla S4 with volume spike AND bearish trend (close < 1d EMA50)
            elif close[i] < s4_1d_aligned[i] and volume_spike[i] and close[i] < ema_50_1d_aligned[i]:
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
2026-05-03 02:32
