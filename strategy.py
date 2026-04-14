#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h/12h Williams Alligator + Volume Spike with 1d Trend Filter
# Uses Williams Alligator (13,8,5 SMAs) on 12h for trend direction, volume confirmation on 6h,
# and 1d EMA50 to filter counter-trend trades. The Alligator's jaw/teeth/lips provide
# dynamic support/resistance. Only trade when price is outside the Alligator's mouth
# (trending) with volume confirmation and aligned with daily trend.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# by fading moves back into the Alligator's mouth during strong trends.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator on 12h: Jaw(13), Teeth(8), Lips(5)
    # All are SMAs of median price (high+low)/2
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values    # Teeth (8)
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values     # Lips (5)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for Alligator and EMA50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Align 12h Alligator lines to 6h
        jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
        teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
        lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
        
        # Align 1d EMA50 to 6h
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        
        price = close[i]
        
        if position == 0:
            # Determine Alligator alignment: check if all lines are ordered
            # For uptrend: Lips > Teeth > Jaw
            # For downtrend: Lips < Teeth < Jaw
            lips_val = lips_aligned[i]
            teeth_val = teeth_aligned[i]
            jaw_val = jaw_aligned[i]
            
            # Long setup: price below Alligator's mouth (Lips) in uptrend with volume spike
            if (lips_val > teeth_val > jaw_val and  # Uptrend alignment
                price < lips_val and                 # Price below mouth
                vol_spike[i] and                     # Volume confirmation
                price > ema_50_aligned[i]):          # Above daily EMA50 (long bias)
                position = 1
                signals[i] = position_size
            # Short setup: price above Alligator's mouth (Lips) in downtrend with volume spike
            elif (lips_val < teeth_val < jaw_val and  # Downtrend alignment
                  price > lips_val and                # Price above mouth
                  vol_spike[i] and                    # Volume confirmation
                  price < ema_50_aligned[i]):         # Below daily EMA50 (short bias)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Alligator's mouth (above Lips) or below daily EMA50
            lips_val = lips_aligned[i]
            if price > lips_val or price < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price re-enters Alligator's mouth (below Lips) or above daily EMA50
            lips_val = lips_aligned[i]
            if price < lips_val or price > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_WilliamsAlligator_Volume_1dEMA50"
timeframe = "6h"
leverage = 1.0