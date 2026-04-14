#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator system with 12-hour trend filter
# Long when: Jaw < Teeth < Lips (bullish alignment) AND price > 12h EMA50 AND Williams %R oversold (< -80)
# Short when: Jaw > Teeth > Lips (bearish alignment) AND price < 12h EMA50 AND Williams %R overbought (> -20)
# Exit when Alligator alignment breaks or price crosses 12h EMA50
# This captures trend continuation after pullbacks in both bull and bear markets
# Target: 80-160 total trades over 4 years (20-40/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12h data ONCE before loop for Alligator and EMA50
    df_12h = get_htf_data(prices, '12h')
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_12h = (high_12h + low_12h) / 2
    jaw_raw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift future data to past
    jaw[:8] = np.nan  # invalidate shifted values
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift future data to past
    teeth[:5] = np.nan  # invalidate shifted values
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift future data to past
    lips[:3] = np.nan  # invalidate shifted values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 12h indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        wr = williams_r_aligned[i]
        ema50 = ema50_12h_aligned[i]
        
        # Bullish alignment: Jaw < Teeth < Lips
        bullish_align = jaw_val < teeth_val < lips_val
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_align = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Long: bullish alignment + above EMA50 + Williams %R oversold
            if bullish_align and price > ema50 and wr < -80:
                position = 1
                signals[i] = position_size
            # Short: bearish alignment + below EMA50 + Williams %R overbought
            elif bearish_align and price < ema50 and wr > -20:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: alignment breaks or price crosses below EMA50
            if not bullish_align or price < ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: alignment breaks or price crosses above EMA50
            if not bearish_align or price > ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_12hEMA50_WR"
timeframe = "6h"
leverage = 1.0