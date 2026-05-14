# Strategy: 12h_Camarilla_R1S1_1dEMA34_VolumeConfirm_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.112 | +11.9% | -17.8% | 74 | FAIL |
| ETHUSDT | 0.103 | +24.1% | -14.8% | 77 | PASS |
| SOLUSDT | 0.756 | +127.0% | -29.4% | 72 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.208 | +8.9% | -9.2% | 25 | PASS |
| SOLUSDT | -0.343 | -3.4% | -18.3% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout + 1d EMA34 trend filter + volume confirmation + ATR trailing stop
# Entry on breakout above Camarilla R1 (long) or below S1 (short) on 12h timeframe.
# 1d EMA34 acts as trend filter: only long when price > EMA34, short when price < EMA34.
# Volume confirmation: current 12h volume > 1.5x 20-period average of 12h volume.
# ATR-based trailing stop (2.5x ATR) to manage risk and reduce whipsaws.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in both bull and bear markets via trend filter and volatility-based stops.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for EMA (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 12h Camarilla Pivot (R1, S1 levels) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point calculation
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    # R1 = pivot + (range * 1.1 / 2)
    r1_12h = pivot_12h + (range_12h * 1.1 / 2)
    # S1 = pivot - (range * 1.1 / 2)
    s1_12h = pivot_12h - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 1d EMA34 (trend filter) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h Volume Confirmation (20-period average) ===
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 12h ATR for trailing stop (14-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema34_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34_val = ema34_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (trend filter reversal) ===
        if position == 1:  # Long position
            # Exit when price crosses below 1d EMA34
            if price < ema34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above 1d EMA34
            if price > ema34_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above R1 AND price > EMA34 AND volume confirmation
            if price > r1_val and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below S1 AND price < EMA34 AND volume confirmation
            elif price < s1_val and price < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_VolumeConfirm_ATRTrail"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-16 21:41
