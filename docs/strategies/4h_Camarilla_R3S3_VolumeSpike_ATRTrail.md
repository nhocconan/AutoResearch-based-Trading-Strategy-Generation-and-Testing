# Strategy: 4h_Camarilla_R3S3_VolumeSpike_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.080 | +15.8% | -11.1% | 106 | FAIL |
| ETHUSDT | 0.144 | +27.3% | -17.6% | 97 | PASS |
| SOLUSDT | 0.779 | +105.9% | -18.6% | 78 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.487 | +13.2% | -7.5% | 32 | PASS |
| SOLUSDT | 0.161 | +7.9% | -11.4% | 24 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 Breakout + Volume Spike + ATR Trailing Stop
# Camarilla pivots from 1d provide institutional support/resistance levels.
# R3/S3 are extreme levels where breakouts often continue with momentum.
# Volume spike (>2x 20-period average) confirms institutional participation.
# ATR trailing stop (3x) manages risk while allowing trends to develop.
# Target: 80-160 total trades over 4 years (20-40/year) to minimize fee drag.
# Works in both bull/bear markets as pivots adapt to volatility and price action.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for Camarilla pivots) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate Camarilla Pivots from 1d data ===
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 as breakout levels
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * daily_range
    camarilla_s3 = close_1d - 1.1 * daily_range
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 4h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # === 4h ATR (15) for trailing stop ===
    atr_4h = np.maximum(high_4h - low_4h, np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), np.abs(low_4h - np.roll(close_4h, 1))))
    atr_4h[0] = high_4h[0] - low_4h[0]  # Fix first value
    atr_ma = pd.Series(atr_4h).ewm(span=15, adjust=False, min_periods=15).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 2.0  # 2x average volume for confirmation
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 3.0*ATR from high
            if price < highest_since_entry - 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 3.0*ATR from low
            if price > lowest_since_entry + 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Camarilla reversal) ===
        if position == 1:  # Long position
            # Exit when price breaks below S3
            if price < s3:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above R3
            if price > r3:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol_confirm:
                # Long when price breaks above R3
                if price > r3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when price breaks below S3
                elif price < s3:
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

name = "4h_Camarilla_R3S3_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 21:17
