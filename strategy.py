#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + 1w Trend Filter
# Use Williams Alligator (Jaw/Teeth/Lips) on 12h to identify trends, volume > 2x median for confirmation,
# and 1-week EMA20 for higher-timeframe trend alignment. Long when Lips > Teeth > Jaw and price above 1w EMA20,
# short when Jaw > Teeth > Lips and price below 1w EMA20. Uses discrete sizing (0.25) to limit overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12-hour Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift by 8 bars
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift by 5 bars
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift by 3 bars
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_vals)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(13, n):  # start after warmup for Alligator
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Lips > Teeth > Jaw (bullish alignment), volume spike, price above 1w EMA20
        if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and
            volume[i] > vol_threshold[i] and close[i] > ema_1w_aligned[i]):
            signals[i] = 0.25
        
        # Short: Jaw > Teeth > Lips (bearish alignment), volume spike, price below 1w EMA20
        elif (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
              volume[i] > vol_threshold[i] and close[i] < ema_1w_aligned[i]):
            signals[i] = -0.25
        
        # Exit: Alligator alignment breaks or price crosses 1w EMA20
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and 
                (lips_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= jaw_aligned[i] or close[i] <= ema_1w_aligned[i])) or
               (signals[i-1] == -0.25 and 
                (jaw_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= lips_aligned[i] or close[i] >= ema_1w_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WilliamsAlligator_Volume_1wEMA"
timeframe = "12h"
leverage = 1.0
EOF