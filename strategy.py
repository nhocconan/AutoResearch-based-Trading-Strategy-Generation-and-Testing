#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and 1d volume confirmation
# Williams Alligator identifies trends via three SMAs (Jaw, Teeth, Lips). 
# When Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend.
# 1w EMA50 ensures we only trade in the weekly trend direction.
# 1d volume spike confirms institutional participation.
# This combination works in both bull and bear markets by filtering for strong trends.
# Targets 12-30 trades per year (~48-120 total over 4 years) to minimize fee drag.

name = "12h_WilliamsAlligator_1wTrend_1dVolume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Williams Alligator on 12h: SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    # Alligator alignment: check if aligned (no overlap)
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    # 1w EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w.values)
    price_above_ema50 = close > ema50_1w_aligned
    price_below_ema50 = close < ema50_1w_aligned
    
    # 1d volume spike (24-period = 12 days approx)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    vol_spike = volume > (vol_ma.values * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (uptrend), price above weekly EMA50, volume spike
            if lips_above_teeth[i] and teeth_above_jaw[i] and price_above_ema50[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (downtrend), price below weekly EMA50, volume spike
            elif lips_below_teeth[i] and teeth_below_jaw[i] and price_below_ema50[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or price crosses weekly EMA50
            if not (lips_above_teeth[i] and teeth_above_jaw[i]) or not price_above_ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or price crosses weekly EMA50
            if not (lips_below_teeth[i] and teeth_below_jaw[i]) or not price_below_ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals