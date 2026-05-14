# Strategy: 6H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_PivotExit_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.557 | +1.2% | -21.4% | 234 | DISCARD |
| ETHUSDT | 0.083 | +23.8% | -15.5% | 216 | KEEP |
| SOLUSDT | 0.848 | +101.8% | -15.4% | 184 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.973 | +19.0% | -6.3% | 67 | KEEP |
| SOLUSDT | 0.977 | +19.5% | -6.1% | 67 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 1.5x 20-period average.
Exit when price retraces to Camarilla pivot point (PP) OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~20-40 trades/year on 6h timeframe.
Camarilla R3/S3 represent strong intraday support/resistance levels; breakouts with volume and trend alignment capture momentum moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from 1d (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = pivot + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = pivot - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        pivot_val = pivot_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend (price > EMA34) AND volume spike (1.5x avg)
            if close[i] > r3_val and close[i] > ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND downtrend (price < EMA34) AND volume spike (1.5x avg)
            elif close[i] < s3_val and close[i] < ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Camarilla pivot point
            if position == 1 and close[i] <= pivot_val:
                exit_signal = True
            elif position == -1 and close[i] >= pivot_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_PivotExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 08:27
