# Strategy: 4h_Camarilla_H4L4_Breakout_12hEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.070 | +15.6% | -13.1% | 146 | FAIL |
| ETHUSDT | 0.693 | +73.6% | -13.2% | 129 | PASS |
| SOLUSDT | 0.512 | +72.5% | -26.7% | 110 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.005 | +23.5% | -7.5% | 43 | PASS |
| SOLUSDT | 0.098 | +6.8% | -11.1% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Camarilla levels from 1d: H4 = close + 1.5*(high-low)/2, L4 = close - 1.5*(high-low)/2.
- Entry: Long when close breaks above H4 AND price > 12h EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when close breaks below L4 AND price < 12h EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: ATR-based trailing stop (2.5 * ATR(14)) from highest high/lowest low since entry.
- Signal size: 0.30 discrete to control fee drag.
- Uses wider Camarilla levels (H4/L4) for fewer, higher-quality breaks; 12h EMA50 provides smoother trend filter than 1d EMA34;
  volume confirmation ensures conviction. Designed to work in both bull (longs) and bear (shorts) markets with controlled drawdown.
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
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Camarilla levels: H4 and L4
    # H4 = close + 1.5*(high-low)/2, L4 = close - 1.5*(high-low)/2
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d) / 2
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma_4h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above H4 AND price > 12h EMA50 (uptrend)
                if curr_close > h4_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Close breaks below L4 AND price < 12h EMA50 (downtrend)
                elif curr_close < l4_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
        elif position == 1:
            # Long position: update highest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.5 * ATR below highest high since entry
            stoploss = highest_since_entry - 2.5 * curr_atr
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: update lowest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.5 * ATR above lowest low since entry
            stoploss = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 18:26
