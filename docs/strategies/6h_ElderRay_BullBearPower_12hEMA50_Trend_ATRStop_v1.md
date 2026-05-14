# Strategy: 6h_ElderRay_BullBearPower_12hEMA50_Trend_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.097 | +24.4% | -17.5% | 132 | PASS |
| ETHUSDT | 0.501 | +53.2% | -14.7% | 108 | PASS |
| SOLUSDT | 1.383 | +280.9% | -22.8% | 120 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.737 | -7.8% | -11.7% | 42 | FAIL |
| ETHUSDT | 0.059 | +6.3% | -10.0% | 34 | PASS |
| SOLUSDT | -0.614 | -3.9% | -14.0% | 36 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA trend filter and ATR-based exits.
- Primary timeframe: 6h to reduce trade frequency and fee drag.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h data).
- Entry: Long when Bull Power > 0 AND Bear Power < 0 AND 12h EMA50 bullish.
         Short when Bear Power < 0 AND Bull Power > 0 AND 12h EMA50 bearish.
         (Note: This condition simplifies to Bull Power > Bear Power for long, Bear Power > Bull Power for short)
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 2.5*ATR,
        exit short when price > lowest_low_since_entry + 2.5*ATR.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
This strategy measures bull/bear power relative to EMA13, filtered by 12h trend to avoid counter-trend trades.
ATR trailing stops allow trends to run while controlling risk. Works in both bull and bear markets
by only taking trades in the direction of the 12h trend, with Elder Power confirming momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate ATR(20) for trailing stop
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)  # Need enough bars for EMA50, EMA13, and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            # Long: Bull Power > Bear Power AND 12h EMA50 bullish (close > EMA)
            if bull_power[i] > bear_power[i] and curr_close > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Bear Power > Bull Power AND 12h EMA50 bearish (close < EMA)
            elif bear_power[i] > bull_power[i] and curr_close < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.5*ATR
            if curr_close < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.5*ATR
            if curr_close > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_ATRStop_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 14:12
