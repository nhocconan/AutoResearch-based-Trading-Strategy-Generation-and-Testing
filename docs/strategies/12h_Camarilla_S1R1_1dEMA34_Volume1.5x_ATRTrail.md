# Strategy: 12h_Camarilla_S1R1_1dEMA34_Volume1.5x_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.082 | +14.1% | -22.1% | 78 | FAIL |
| ETHUSDT | 0.174 | +29.6% | -17.7% | 75 | PASS |
| SOLUSDT | 0.786 | +131.5% | -24.8% | 75 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.008 | +4.9% | -11.5% | 28 | PASS |
| SOLUSDT | 0.271 | +10.5% | -17.2% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot (S1/R1) breakout with 1d EMA34 filter and volume confirmation
# Long when price breaks above 1d Camarilla R1 AND price > 1d EMA34 AND volume > 1.5x 24-period average volume
# Short when price breaks below 1d Camarilla S1 AND price < 1d EMA34 AND volume > 1.5x 24-period average volume
# ATR trailing stop (2x ATR) to manage risk
# Camarilla pivots provide precise intraday levels with statistical edge
# EMA34 filter ensures trend alignment, reducing counter-trend trades
# Volume confirmation adds conviction to breakouts
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA34 filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1d Camarilla Pivot levels (S1, R1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h Volume Confirmation (24-period average = 12 days) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # === 12h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
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
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_24[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_confirm = volume[i] > vol_ma_24[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Camarilla R1 AND price > EMA34 AND volume confirmation
            if price > r1_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Camarilla S1 AND price < EMA34 AND volume confirmation
            elif price < s1_val and price < ema_val and vol_confirm:
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

name = "12h_Camarilla_S1R1_1dEMA34_Volume1.5x_ATRTrail"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-16 22:45
