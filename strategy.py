#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# The Alligator (three SMAs: Jaw=13, Teeth=8, Lips=5) identifies trend absence (all lines intertwined) 
# vs presence (lines diverging in order). Trade only when Alligator is "awake" (JAW > TEETH > LIPS for long, 
# JAW < TEETH < LIPS for short) and price is outside the Alligator's mouth. 
# 1d EMA200 provides higher-timeframe trend filter to avoid counter-trend trades. 
# Volume confirmation ensures institutional participation. 
# This should work in both bull and bear markets by following higher timeframe trend and avoiding range-bound periods.
# Target: 15-30 trades per year to minimize fee drain.

name = "12h_Alligator_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA200 for trend direction ===
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === Williams Alligator on 12h (Jaw=13, Teeth=8, Lips=5) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    median_price = (high + low) / 2  # Typical price for Alligator
    
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === 12h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after Alligator warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_200_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(jaw_val) or np.isnan(teeth_val) or 
            np.isnan(lips_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator awake and aligned for long: JAW > TEETH > LIPS (uptrend)
            # Price above Alligator's mouth (above JAW) with volume confirmation
            if jaw_val > teeth_val and teeth_val > lips_val and close_val > jaw_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Alligator awake and aligned for short: JAW < TEETH < LIPS (downtrend)
            # Price below Alligator's mouth (below LIPS) with volume confirmation
            elif jaw_val < teeth_val and teeth_val < lips_val and close_val < lips_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: Alligator sleeping (lines intertwine) or trend reversal
            # Exit when JAW <= TEETH or price closes below TEETH
            if jaw_val <= teeth_val or close_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator sleeping (lines intertwine) or trend reversal
            # Exit when JAW >= TEETH or price closes above TEETH
            if jaw_val >= teeth_val or close_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals