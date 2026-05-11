#!/usr/bin/env python3
# 6h_ElderRay_Alligator_Combination
# Hypothesis: Combines Elder Ray (Bull/Bear Power) with Williams Alligator on 1d timeframe to identify trend strength and direction.
# Elder Ray measures bull/bear power relative to EMA13; Alligator uses SMAs (13,8,5) to define trend.
# Long when Bull Power > 0, Bear Power < 0, and price above Alligator's Jaw (red line).
# Short when Bear Power > 0, Bull Power < 0, and price below Alligator's Jaw.
# Works in bull markets by catching strong uptrends and in bear markets by identifying sustained downtrends.
# Uses volume confirmation to avoid false signals and reduce whipsaws.

name = "6h_ElderRay_Alligator_Combination"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Elder Ray and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Elder Ray: Bull Power and Bear Power ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # --- 1d Alligator: Jaw (13), Teeth (8), Lips (5) SMAs ---
    sma13_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Jaw
    sma8_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values      # Teeth
    sma5_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values      # Lips
    
    # Align all 1d indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, sma13_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, sma8_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, sma5_1d)
    
    # --- Volume confirmation (volume > 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA13 (13) and SMA13 (13)
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(jaw_1d_aligned[i]) or
            np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        
        # Alligator alignment: check if jaws, teeth, lips are properly aligned
        # In uptrend: Lips > Teeth > Jaw (green, red, blue from bottom)
        # In downtrend: Jaw > Teeth > Lips (blue, red, green from top)
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        
        # Uptrend alignment: Lips > Teeth > Jaw
        uptrend_aligned = lips > teeth and teeth > jaw
        # Downtrend alignment: Jaw > Teeth > Lips
        downtrend_aligned = jaw > teeth and teeth > lips
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price above Jaw, volume surge, uptrend alignment
            if bull_power > 0 and bear_power < 0 and close[i] > jaw and vol_surge[i] and uptrend_aligned:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0, price below Jaw, volume surge, downtrend alignment
            elif bear_power > 0 and bull_power < 0 and close[i] < jaw and vol_surge[i] and downtrend_aligned:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: Bear Power becomes positive OR price crosses below Jaw
                if bear_power >= 0 or close[i] < jaw:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Bull Power becomes positive OR price crosses above Jaw
                if bull_power >= 0 or close[i] > jaw:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals