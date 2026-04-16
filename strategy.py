#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and 12h Fractal filter
# Long when: Green line (Jaw) > Red line (Teeth) > Blue line (Lips) AND price > Green line AND volume > 1.5x 1d average
# Short when: Green line < Red line < Blue line AND price < Green line AND volume > 1.5x 1d average
# Williams Alligator uses SMAs of median price with specific offsets to identify trend alignment
# Williams Alligator is effective in both trending and ranging markets when combined with volume confirmation
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Williams Alligator (Jaw, Teeth, Lips) ===
    df_12h = get_htf_data(prices, '12h')
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Jaw (Blue line): 13-period SMMA, shifted 8 bars
    jaw_raw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean()
    jaw = np.roll(jaw_raw.values, 8)
    jaw[:8] = np.nan
    
    # Teeth (Red line): 8-period SMMA, shifted 5 bars
    teeth_raw = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean()
    teeth = np.roll(teeth_raw.values, 5)
    teeth[:5] = np.nan
    
    # Lips (Green line): 5-period SMMA, shifted 3 bars
    lips_raw = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean()
    lips = np.roll(lips_raw.values, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # === 1d Volume Confirmation ===
    vol_ma_1d = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods of 1h = 1d (4h data)
    
    # === 4h ATR for stop management ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr[i]
        
        # === STOP LOSS LOGIC (2x ATR) ===
        if position == 1:  # Long position
            if price < lips_val - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if price > lips_val + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Alligator aligned for uptrend: Jaw > Teeth > Lips
            if jaw_val > teeth_val and teeth_val > lips_val and price > lips_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Alligator aligned for downtrend: Jaw < Teeth < Lips
            elif jaw_val < teeth_val and teeth_val < lips_val and price < lips_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_12hMedianPrice_1dVolume1.5x"
timeframe = "4h"
leverage = 1.0