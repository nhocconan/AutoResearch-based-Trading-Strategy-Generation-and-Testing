# Strategy: 4h_Donchian20_1dEMA200_Volume2x_ATRTrail_2.5x

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.394 | +36.4% | -7.9% | 78 | PASS |
| ETHUSDT | 0.544 | +48.1% | -12.8% | 68 | PASS |
| SOLUSDT | 1.322 | +174.5% | -14.0% | 61 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.949 | -2.2% | -10.3% | 32 | FAIL |
| ETHUSDT | 0.215 | +8.7% | -12.6% | 32 | PASS |
| SOLUSDT | -0.007 | +5.2% | -9.2% | 29 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper(20) AND price > 1d EMA200 AND volume > 2x 24-period avg volume
# Short when price breaks below 4h Donchian lower(20) AND price < 1d EMA200 AND volume > 2x 24-period avg volume
# ATR trailing stop (2.5x ATR) to manage risk
# Donchian provides clear trend-following structure
# EMA200 filter ensures alignment with daily trend, reducing counter-trend trades
# Volume confirmation adds conviction to breakouts
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA200 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === 4h Donchian channels (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Upper = max(high, lookback 20), Lower = min(low, lookback 20)
    donch_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    
    # === 4h Volume Confirmation (24-period average = 6 hours) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # === 4h ATR for trailing stop (15-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=15, min_periods=15).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(donch_upper_aligned[i]) or
            np.isnan(donch_lower_aligned[i]) or
            np.isnan(vol_ma_24[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_200_aligned[i]
        upper_val = donch_upper_aligned[i]
        lower_val = donch_lower_aligned[i]
        vol_confirm = volume[i] > vol_ma_24[i] * 2.0  # 2x average volume for confirmation
        atr_val = atr[i]
        
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
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian upper AND price > EMA200 AND volume confirmation
            if price > upper_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Donchian lower AND price < EMA200 AND volume confirmation
            elif price < lower_val and price < ema_val and vol_confirm:
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

name = "4h_Donchian20_1dEMA200_Volume2x_ATRTrail_2.5x"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 22:47
