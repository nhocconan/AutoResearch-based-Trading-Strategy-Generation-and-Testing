# Strategy: 4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop_PPExit

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.434 | +37.7% | -6.1% | 250 | PASS |
| ETHUSDT | 0.322 | +35.8% | -9.8% | 234 | PASS |
| SOLUSDT | 0.728 | +83.0% | -17.3% | 201 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.389 | -4.5% | -6.6% | 97 | FAIL |
| ETHUSDT | 1.039 | +20.3% | -5.8% | 85 | PASS |
| SOLUSDT | 0.680 | +14.7% | -6.3% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above 1d Camarilla R1 AND close > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below 1d Camarilla S1 AND close < 1d EMA34 AND volume > 2.0x 20-period average.
Exit when price retraces to 1d Camarilla pivot point (PP) or ATR trailing stop hit (2.5*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
Targets 20-40 trades/year per symbol (80-160 total over 4 years) to avoid fee drag.
Designed for BTC and ETH as primary targets with strict entry conditions to filter false breakouts.
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
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, Range = H-L
    camarilla_pp = (h_1d + l_1d + c_1d) / 3.0
    camarilla_range = h_1d - l_1d
    camarilla_r1 = camarilla_pp + (camarilla_range * 1.0 / 12.0)
    camarilla_s1 = camarilla_pp - (camarilla_range * 1.0 / 12.0)
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(c_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
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
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND uptrend (close > EMA34) AND volume spike
            if price > r1_val and close[i] > ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price breaks below Camarilla S1 AND downtrend (close < EMA34) AND volume spike
            elif price < s1_val and close[i] < ema34_val and volume[i] > 2.0 * vol_ma_val:
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
            
            # Primary exit: Price retraces to Camarilla pivot point (PP)
            if position == 1 and price <= pp_val:
                exit_signal = True
            elif position == -1 and price >= pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop_PPExit"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 06:35
