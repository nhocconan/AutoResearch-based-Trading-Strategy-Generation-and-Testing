# Strategy: 6h_ATRBreakout_1dEMA34_1wEMA21_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.535 | +50.2% | -9.2% | 54 | PASS |
| ETHUSDT | 0.399 | +45.2% | -14.7% | 50 | PASS |
| SOLUSDT | 1.031 | +166.4% | -22.1% | 59 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.071 | -1.4% | -6.2% | 19 | FAIL |
| ETHUSDT | 0.211 | +8.5% | -8.1% | 13 | PASS |
| SOLUSDT | 0.155 | +7.7% | -7.7% | 13 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 1d data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Get 1w data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # 1w EMA(21) for higher timeframe trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Get 6h data for breakout signals (ATR-based breakout)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    open_6h = df_6h['open'].values
    
    # 6h ATR(14) for breakout threshold
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = tr2_6h[0] = tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h_14 = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Breakout threshold: 0.5 * ATR(14) from open
    upper_breakout = open_6h + 0.5 * atr_6h_14
    lower_breakout = open_6h - 0.5 * atr_6h_14
    upper_breakout_aligned = align_htf_to_ltf(prices, df_6h, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_6h, lower_breakout)
    
    # Volume confirmation: current volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA (primary) and 1w EMA (higher timeframe)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        htf_uptrend = close[i] > ema_21_1w_aligned[i]
        htf_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_aligned[i] > np.mean(atr_14_aligned[max(0, i-50):i+1]) * 0.8
        
        # Breakout conditions: price breaks 0.5*ATR from 6h open
        long_breakout = close[i] > upper_breakout_aligned[i]
        short_breakout = close[i] < lower_breakout_aligned[i]
        
        # Entry conditions: require alignment of 1d and 1w trends
        long_entry = long_breakout and uptrend and htf_uptrend and vol_filter and volume_confirm[i]
        short_entry = short_breakout and downtrend and htf_downtrend and vol_filter and volume_confirm[i]
        
        # Exit conditions: reverse signal or volatility collapse
        if position == 1:
            exit_condition = not uptrend or not htf_uptrend or (atr_14_aligned[i] < np.mean(atr_14_aligned[max(0, i-20):i+1]) * 0.5)
        elif position == -1:
            exit_condition = not downtrend or not htf_downtrend or (atr_14_aligned[i] < np.mean(atr_14_aligned[max(0, i-20):i+1]) * 0.5)
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

name = "6h_ATRBreakout_1dEMA34_1wEMA21_VolumeFilter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-28 09:47
