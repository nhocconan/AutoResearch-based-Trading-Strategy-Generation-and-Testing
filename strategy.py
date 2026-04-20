#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray + Volume Filter
# Uses Alligator (Jaw/Teeth/Lips) to detect trend direction and strength.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# Enters long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND Volume > 1.5x average.
# Enters short when: Jaw > Teeth > Lips (bearish alignment) AND Bear Power > 0 AND Volume > 1.5x average.
# Exits when Alligator alignment breaks or power weakens.
# Alligator avoids whipsaws in ranging markets, Elder Ray confirms momentum, Volume ensures conviction.
# Weekly trend filter (price > weekly EMA20 for long, < for short) avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Alligator_ElderRay_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Williams Alligator (13,8,5 SMAs shifted) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # === Elder Ray (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # === Weekly EMA20 for trend filter ===
    weekly_close = df_1w['close'].values
    ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Start after Alligator warmup (max shift 8 + 13)
        # Get values
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        ema_val = ema_20_aligned[i]
        vol_ratio_val = vol_ratio[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(bull_power_val) or np.isnan(bear_power_val) or 
            np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bullish Alligator alignment + Bull Power > 0 + Volume + Weekly uptrend
            if (lips_val > teeth_val > jaw_val and 
                bull_power_val > 0 and 
                vol_ratio_val > 1.5 and 
                close_val > ema_val):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator alignment + Bear Power > 0 + Volume + Weekly downtrend
            elif (jaw_val > teeth_val > lips_val and 
                  bear_power_val > 0 and 
                  vol_ratio_val > 1.5 and 
                  close_val < ema_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks OR Bull Power weakens
            if not (lips_val > teeth_val > jaw_val) or bull_power_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks OR Bear Power weakens
            if not (jaw_val > teeth_val > lips_val) or bear_power_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals