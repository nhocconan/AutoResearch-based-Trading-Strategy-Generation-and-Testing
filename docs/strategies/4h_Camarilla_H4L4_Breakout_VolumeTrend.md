# Strategy: 4h_Camarilla_H4L4_Breakout_VolumeTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.311 | +32.0% | -4.5% | 419 | PASS |
| ETHUSDT | 0.042 | +22.0% | -9.1% | 415 | PASS |
| SOLUSDT | 1.035 | +108.2% | -13.2% | 405 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.758 | -4.8% | -7.7% | 140 | FAIL |
| ETHUSDT | 0.681 | +11.3% | -2.6% | 75 | PASS |
| SOLUSDT | -0.006 | +5.9% | -5.2% | 86 | FAIL |

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
    
    # Get 1d data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (H4/L4 for entries, H3/L3 for exits)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    H4 = close_1d + (range_hl * 1.1 / 2)
    L4 = close_1d - (range_hl * 1.1 / 2)
    H3 = close_1d + (range_hl * 1.1 / 4)
    L3 = close_1d - (range_hl * 1.1 / 4)
    
    # Align pivot levels to 4h
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get 4h data for volume and volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Volume ratio (current 4h volume / 20-period average)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # ATR(14) for volatility filter
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current 4h volume above average
        volume_filter = volume_4h[i] > vol_ma_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_4h_aligned[i] > 0.001 * close[i]  # At least 0.1% ATR
        
        # Entry conditions: Camarilla H4/L4 breakout with volume and trend
        long_breakout = close[i] > H4_aligned[i]
        short_breakout = close[i] < L4_aligned[i]
        
        long_entry = uptrend and long_breakout and volume_filter and vol_filter
        short_entry = downtrend and short_breakout and volume_filter and vol_filter
        
        # Exit conditions: Camarilla H3/L3 retracement
        long_exit = close[i] < H3_aligned[i]
        short_exit = close[i] > L3_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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

name = "4h_Camarilla_H4L4_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 10:47
