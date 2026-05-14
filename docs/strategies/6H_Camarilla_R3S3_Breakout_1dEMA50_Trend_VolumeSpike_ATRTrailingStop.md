# Strategy: 6H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.019 | +20.1% | -10.3% | 315 | FAIL |
| ETHUSDT | 0.403 | +39.3% | -6.0% | 294 | PASS |
| SOLUSDT | 0.741 | +86.4% | -20.5% | 261 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.436 | +11.4% | -8.3% | 104 | PASS |
| SOLUSDT | -0.458 | -0.7% | -14.7% | 101 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter, volume confirmation, and ATR trailing stop.
Long when price breaks above 12h Camarilla R3 level AND price > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below 12h Camarilla S3 level AND price < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when price retraces to 12h Camarilla Pivot (midpoint) or ATR trailing stop hit (2.5*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to reduce trade frequency vs previous versions.
Camarilla R3/S3 from 12h timeframe provides fewer but higher-quality breakouts suitable for 6h entries.
Designed for 6h timeframe targeting ~12-25 trades/year per symbol (50-100 total over 4 years).
Focus on BTC and ETH as primary targets with SOL as secondary confirmation.
"""

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
    
    # Calculate 12h Camarilla pivot levels (R3, S3, Pivot)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous 12h bar's high, low, close
    # R3 = Close + (High - Low) * 1.125/4
    # S3 = Close - (High - Low) * 1.125/4
    # Pivot = (High + Low + Close) / 3
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    camarilla_r3 = c_12h + (h_12h - l_12h) * 1.125 / 4.0
    camarilla_s3 = c_12h - (h_12h - l_12h) * 1.125 / 4.0
    camarilla_pivot = (h_12h + l_12h + c_12h) / 3.0
    
    # Align 12h Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # Camarilla needs 1, EMA needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        pivot_val = camarilla_pivot_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 12h Camarilla R3 AND price > 1d EMA50 AND volume spike
            if (price > r3_val and price > ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 12h Camarilla S3 AND price < 1d EMA50 AND volume spike
            elif (price < s3_val and price < ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 12h Camarilla Pivot (midpoint)
            if position == 1 and price <= pivot_val:
                exit_signal = True
            elif position == -1 and price >= pivot_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 06:27
