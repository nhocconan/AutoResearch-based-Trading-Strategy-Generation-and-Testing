# Strategy: 6h_ElderRay_12hEMA50_VolumeConfirm_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.062 | +22.1% | -15.7% | 224 | PASS |
| ETHUSDT | 0.128 | +26.0% | -19.4% | 219 | PASS |
| SOLUSDT | 1.080 | +217.7% | -33.6% | 209 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.808 | -4.4% | -9.4% | 81 | FAIL |
| ETHUSDT | 0.185 | +8.5% | -12.3% | 78 | PASS |
| SOLUSDT | -0.180 | +0.5% | -12.5% | 74 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation
# Long when Bull Power > 0, price > 12h EMA50, and volume > 1.5x 20-period average
# Short when Bear Power < 0, price < 12h EMA50, and volume > 1.5x 20-period average
# ATR-based trailing stop (2.0x ATR) to manage risk and reduce whipsaws
# Elder Ray measures bull/bear strength relative to EMA, effective in both trending and ranging markets
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h EMA50 (trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 12h EMA13 (for Elder Ray calculation) ===
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    
    # === 6h ATR for trailing stop (14-period) ===
    df_6h = get_htf_data(prices, '6h')
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
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
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
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_13_12h_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_13_val = ema_13_12h_aligned[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        atr_val = atr_aligned[i]
        
        # Calculate Elder Ray components
        bull_power = high[i] - ema_13_val  # Bull Power: High - EMA13
        bear_power = low[i] - ema_13_val   # Bear Power: Low - EMA13
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (trend filter reversal) ===
        if position == 1:  # Long position
            # Exit when price crosses below 12h EMA50
            if price < ema_50_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above 12h EMA50
            if price > ema_50_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Bull Power > 0 AND price > 12h EMA50 AND volume confirmation
            if bull_power > 0 and price > ema_50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: Bear Power < 0 AND price < 12h EMA50 AND volume confirmation
            elif bear_power < 0 and price < ema_50_val and vol_confirm:
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

name = "6h_ElderRay_12hEMA50_VolumeConfirm_ATRTrail"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 22:04
