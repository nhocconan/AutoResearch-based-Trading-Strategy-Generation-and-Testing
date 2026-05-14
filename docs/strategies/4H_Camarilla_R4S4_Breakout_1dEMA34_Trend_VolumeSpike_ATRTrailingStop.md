# Strategy: 4H_Camarilla_R4S4_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.103 | +16.5% | -8.1% | 309 | FAIL |
| ETHUSDT | 0.132 | +26.3% | -8.9% | 284 | PASS |
| SOLUSDT | 0.715 | +87.2% | -14.3% | 263 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.054 | +6.3% | -7.9% | 93 | PASS |
| SOLUSDT | 0.747 | +16.8% | -8.0% | 87 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla R4/S4 breakout with 1d EMA34 trend filter, volume confirmation, and ATR trailing stop.
Long when price breaks above 1d Camarilla R4 level AND price > 1d EMA34 AND volume > 1.6x 20-period average.
Short when price breaks below 1d Camarilla S4 level AND price < 1d EMA34 AND volume > 1.6x 20-period average.
Exit when price retraces to 1d Camarilla Pivot (midpoint) or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to reduce trade frequency vs previous version.
Camarilla R4/S4 are more extreme levels (1.5/4 * range) than R3/S3, providing fewer but higher-quality breakouts.
Designed for 4h timeframe targeting ~25 trades/year per symbol (100 total over 4 years).
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
    
    # Calculate 1d Camarilla pivot levels (R4, S4, Pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's high, low, close
    # R4 = Close + (High - Low) * 1.5/4
    # S4 = Close - (High - Low) * 1.5/4
    # Pivot = (High + Low + Close) / 3
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r4 = c_1d + (h_1d - l_1d) * 1.5 / 4.0
    camarilla_s4 = c_1d - (h_1d - l_1d) * 1.5 / 4.0
    camarilla_pivot = (h_1d + l_1d + c_1d) / 3.0
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 1d EMA34 for trend filter
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume average (20-period) on 4h timeframe
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
    start_idx = max(1, 34, 20)  # Camarilla needs 1, EMA needs 34, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        pivot_val = camarilla_pivot_aligned[i]
        ema_34_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla R4 AND price > 1d EMA34 AND volume spike
            if (price > r4_val and price > ema_34_val and volume[i] > 1.6 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1d Camarilla S4 AND price < 1d EMA34 AND volume spike
            elif (price < s4_val and price < ema_34_val and volume[i] > 1.6 * vol_ma_val):
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
            
            # Primary exit: Price retraces to 1d Camarilla Pivot (midpoint)
            if position == 1 and price <= pivot_val:
                exit_signal = True
            elif position == -1 and price >= pivot_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
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

name = "4H_Camarilla_R4S4_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 06:23
