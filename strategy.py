#!/usr/bin/env python3
"""
12h_Williams_Alligator_Trend
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) trend filter with 1w HTF direction and volume confirmation. Long when Lips > Teeth > Jaw and price > Lips with volume > 1.5x 20-bar mean; Short when Lips < Teeth < Jaw and price < Lips with volume confirmation. Uses discrete sizing (0.25) to minimize fees. Designed for 12-25 trades/year per symbol, effective in trending markets (bull/bear) by capturing medium-term momentum with Alligator's smoothed averages reducing whipsaw.
"""

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
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Williams Alligator on 12h: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
    median_12h = (high_12h + low_12h) / 2
    jaw_12h = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_12h = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_12h = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA(34) for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Volume confirmation: current volume > 1.5x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (13+8=21) and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Bullish Alligator: Lips > Teeth > Jaw
            bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            # Bearish Alligator: Lips < Teeth < Jaw
            bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            
            # Long: bullish Alligator, price > Lips, HTF uptrend (price > 1w EMA34), volume confirmation
            long_signal = bullish and (close[i] > lips_aligned[i]) and (close[i] > ema_34_1w_aligned[i]) and vol_confirm[i]
            # Short: bearish Alligator, price < Lips, HTF downtrend (price < 1w EMA34), volume confirmation
            short_signal = bearish and (close[i] < lips_aligned[i]) and (close[i] < ema_34_1w_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when Alligator turns bearish OR price < Lips
            exit_signal = (lips_aligned[i] < teeth_aligned[i]) or (close[i] < lips_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Alligator turns bullish OR price > Lips
            exit_signal = (lips_aligned[i] > teeth_aligned[i]) or (close[i] > lips_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Williams_Alligator_Trend"
timeframe = "12h"
leverage = 1.0