# Strategy: 4h_TrendBreakout_ATRStop_1dEMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.337 | +1.7% | -13.7% | 435 | FAIL |
| ETHUSDT | 0.251 | +35.4% | -14.3% | 440 | PASS |
| SOLUSDT | 0.728 | +113.7% | -23.3% | 421 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.103 | +27.4% | -11.1% | 135 | PASS |
| SOLUSDT | 1.398 | +36.5% | -7.5% | 128 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for ATR calculation (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(14) on 4h
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h = np.concatenate([[np.nan], tr_4h])  # Align with original index
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions: trend following with volatility filter
        long_entry = uptrend and close[i] > high[i-1] and volume[i] > 1.2 * np.nanmean(volume[max(0,i-5):i]) if i >= 5 else False
        short_entry = downtrend and close[i] < low[i-1] and volume[i] > 1.2 * np.nanmean(volume[max(0,i-5):i]) if i >= 5 else False
        
        # Exit conditions: ATR-based stop loss
        if position == 1:
            # Trail stop: exit if price drops 2*ATR from highest high since entry
            # We'll use a simple trailing stop based on recent high
            recent_high = np.nanmax(high[max(0,i-10):i+1]) if i >= 10 else high[i]
            exit_condition = close[i] < recent_high - 2.0 * atr_4h_aligned[i]
        elif position == -1:
            # Trail stop: exit if price rises 2*ATR from lowest low since entry
            recent_low = np.nanmin(low[max(0,i-10):i+1]) if i >= 10 else low[i]
            exit_condition = close[i] > recent_low + 2.0 * atr_4h_aligned[i]
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_TrendBreakout_ATRStop_1dEMA34"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 10:28
