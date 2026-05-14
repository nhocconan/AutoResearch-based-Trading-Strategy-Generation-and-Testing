# Strategy: 6h_Camarilla_R1S1_12hEMA34_VolumeConfirm_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.058 | +15.1% | -14.6% | 162 | FAIL |
| ETHUSDT | 0.320 | +41.9% | -16.1% | 158 | PASS |
| SOLUSDT | 1.172 | +242.3% | -24.2% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.453 | +14.0% | -12.1% | 52 | PASS |
| SOLUSDT | 0.627 | +18.6% | -12.5% | 49 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot R1/S1 breakout with 12h EMA34 trend filter and volume confirmation
# Long when price breaks above R1 (1.0833 * (H-L) + C) with price > 12h EMA34 and volume > 1.5x 20-period average
# Short when price breaks below S1 (C - 1.0833 * (H-L)) with price < 12h EMA34 and volume > 1.5x 20-period average
# ATR-based trailing stop (2.5x ATR) to manage risk and reduce whipsaws
# Camarilla pivots derived from prior 6h bar (H,L,C) to avoid look-ahead
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Works in both bull and bear markets via trend filter and volatility-based stops

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data for Camarilla pivots, volume, ATR ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === Camarilla Pivot R1 and S1 from prior 6h bar ===
    # R1 = Close + 1.0833 * (High - Low)
    # S1 = Close - 1.0833 * (High - Low)
    # Using prior bar values to avoid look-ahead
    prior_high_6h = np.roll(high_6h, 1)
    prior_low_6h = np.roll(low_6h, 1)
    prior_close_6h = np.roll(close_6h, 1)
    prior_high_6h[0] = high_6h[0]  # first bar uses current values
    prior_low_6h[0] = low_6h[0]
    prior_close_6h[0] = close_6h[0]
    
    camarilla_r1_6h = prior_close_6h + 1.0833 * (prior_high_6h - prior_low_6h)
    camarilla_s1_6h = prior_close_6h - 1.0833 * (prior_high_6h - prior_low_6h)
    
    r1_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r1_6h)
    s1_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s1_6h)
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # === 6h ATR for trailing stop (14-period) ===
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # === 12h EMA34 (trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
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
            # Exit when price crosses below 12h EMA34
            if price < ema34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above 12h EMA34
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

name = "6h_Camarilla_R1S1_12hEMA34_VolumeConfirm_ATRTrail"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 21:53
