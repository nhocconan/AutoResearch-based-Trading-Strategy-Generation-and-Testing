#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with weekly trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (all lines intertwined) vs presence (diverged lines).
# Weekly EMA trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation filters low-conviction breakouts.
# Works in bull/bear by using weekly EMA trend filter (long only above weekly EMA, short only below weekly EMA)
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d timeframe (smoothed medians)
    df_1d = get_htf_data(prices, '1d')
    median_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8).values
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5).values
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(24, 13+8, 8+5, 5+3)  # for volume average and Alligator shifts
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND above weekly EMA50 AND volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and
                price > lips_aligned[i] and price > ema_50_1w_aligned[i] and
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: Jaws > Teeth > Lips (bearish alignment) AND price < Jaws AND below weekly EMA50 AND volume confirmation
            elif (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
                  price < jaw_aligned[i] and price < ema_50_1w_aligned[i] and
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Lips < Teeth OR price < Teeth OR below weekly EMA50
            if (lips_aligned[i] < teeth_aligned[i] or price < teeth_aligned[i] or
                price < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Jaws < Teeth OR price > Teeth OR above weekly EMA50
            if (jaw_aligned[i] < teeth_aligned[i] or price > teeth_aligned[i] or
                price > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Williams_Alligator_WeeklyEMA_Volume"
timeframe = "12h"
leverage = 1.0