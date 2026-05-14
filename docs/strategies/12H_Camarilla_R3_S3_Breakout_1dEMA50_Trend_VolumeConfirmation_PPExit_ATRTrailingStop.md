# Strategy: 12H_Camarilla_R3_S3_Breakout_1dEMA50_Trend_VolumeConfirmation_PPExit_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.468 | +5.0% | -14.2% | 95 | FAIL |
| ETHUSDT | 0.069 | +23.2% | -12.8% | 87 | PASS |
| SOLUSDT | 0.113 | +24.0% | -37.0% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.333 | +10.4% | -7.3% | 38 | PASS |
| SOLUSDT | -0.889 | -6.1% | -17.6% | 34 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 1d EMA50 AND volume > 1.8x 24-period average.
Short when price breaks below Camarilla S3 AND close < 1d EMA50 AND volume > 1.8x 24-period average.
Exit when price retraces to Camarilla pivot point or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.30) and volume filter to target 12-37 trades/year.
12h timeframe reduces noise and fee drag vs lower timeframes while capturing multi-day trends.
"""

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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from 1d timeframe (R3/S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume average (24-period for 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
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
    start_idx = max(50, 2, 24)  # EMA50 needs 50, Camarilla needs 2, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        pp = pp_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend (price > EMA50) AND volume spike (1.8x avg)
            if close[i] > r3 and close[i] > ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.30
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND downtrend (price < EMA50) AND volume spike (1.8x avg)
            elif close[i] < s3 and close[i] < ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.30
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
            if position == 1 and close[i] <= pp:
                exit_signal = True
            elif position == -1 and close[i] >= pp:
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
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_Camarilla_R3_S3_Breakout_1dEMA50_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 07:31
