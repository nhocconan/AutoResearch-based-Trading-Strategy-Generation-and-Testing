# Strategy: 6h_Donchian20_1dEMA34_VolumeConfirm_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.110 | +25.1% | -16.2% | 113 | PASS |
| ETHUSDT | 0.142 | +27.2% | -12.0% | 112 | PASS |
| SOLUSDT | 0.970 | +160.3% | -20.1% | 100 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.123 | -6.2% | -9.4% | 41 | FAIL |
| ETHUSDT | 0.301 | +10.4% | -7.6% | 34 | PASS |
| SOLUSDT | 0.542 | +15.7% | -9.7% | 32 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + ATR trailing stop
# Entry on breakout above Donchian upper (long) or below Donchian lower (short) on 6h timeframe.
# 1d EMA34 acts as trend filter: only long when price > EMA34, short when price < EMA34.
# Volume confirmation: current 6h volume > 1.5x 20-period average of 6h volume.
# ATR-based trailing stop (2.5x ATR) to manage risk and reduce whipsaws.
# Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag.
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
    
    # === 6h Donchian Channel (20-period) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Upper band: highest high over last 20 periods
    upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (already aligned via get_htf_data)
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    
    # === 1d EMA34 (trend filter) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 6h Volume Confirmation (20-period average) ===
    volume_6h = df_6h['volume'].values
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # === 6h ATR for trailing stop (14-period) ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
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
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(ema34_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
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
            # Long when: price breaks above upper band AND price > EMA34 AND volume confirmation
            if price > upper_val and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below lower band AND price < EMA34 AND volume confirmation
            elif price < lower_val and price < ema34_val and vol_confirm:
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

name = "6h_Donchian20_1dEMA34_VolumeConfirm_ATRTrail"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 21:41
