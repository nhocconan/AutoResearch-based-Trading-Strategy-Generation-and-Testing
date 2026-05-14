# Strategy: 6h_Alligator_WilliamsR_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.050 | +22.1% | -19.0% | 215 | PASS |
| ETHUSDT | 0.237 | +34.0% | -21.5% | 203 | PASS |
| SOLUSDT | -1.410 | -67.0% | -68.0% | 199 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.091 | +6.8% | -6.4% | 71 | PASS |
| ETHUSDT | -0.978 | -12.2% | -21.1% | 73 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Williams Alligator (Jaws, Teeth, Lips) ===
    # Jaws: SMA(13) of median price, shifted 8 bars forward
    median_price_1d = (high_1d + low_1d) / 2
    jaws = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8) of median price, shifted 5 bars forward
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5) of median price, shifted 3 bars forward
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components with proper delay for forward shifts
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 6h Williams %R for momentum (14 period) ===
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Alligator and Williams %R
    warmup = 30
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ratio_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        lip = lips_aligned[i]
        wr = williams_r_aligned[i]
        vol_ratio = vol_ratio_6h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: Williams %R overbought or Alligator lines cross bearish
            if wr > -20 or (jaw < tooth < lip):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: Williams %R oversold or Alligator lines cross bullish
            if wr < -80 or (jaw > tooth > lip):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Alligator alignment: bullish (jaw > tooth > lip) or bearish (jaw < tooth < lip)
            # Williams %R: not extreme (> -80 and < -20) to avoid chop
            # Volume confirmation: above average
            if jaw > tooth > lip and wr > -80 and wr < -20 and vol_ratio > 1.2:
                # LONG: Alligator bullish, Williams not oversold, volume confirmation
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            elif jaw < tooth < lip and wr > -80 and wr < -20 and vol_ratio > 1.2:
                # SHORT: Alligator bearish, Williams not overbought, volume confirmation
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Alligator_WilliamsR_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 18:28
