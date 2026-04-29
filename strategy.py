#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + Regime Filter
# Williams Alligator uses three SMAs (jaw=13, teeth=8, lips=5) to identify trends.
# Long when lips > teeth > jaw (bullish alignment) AND volume > 2.0x 20-bar avg AND price > 1d EMA50
# Short when lips < teeth < jaw (bearish alignment) AND volume > 2.0x 20-bar avg AND price < 1d EMA50
# Exit when Alligator alignment breaks (lips crosses teeth) OR price crosses 8-bar SMA
# Uses discrete position sizing (0.25) to reduce fee drag.
# Alligator identifies trend presence/absence, volume confirmation ensures follow-through,
# 1d EMA50 filters counter-trend moves. Works in trending markets (Alligator awake) and avoids chop.

name = "12h_WilliamsAlligator_VolumeSpike_1dEMA50_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator SMAs on 12h data
    close_s = pd.Series(close)
    # Jaw: 13-period SMA (slowest)
    jaw = close_s.rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMA (middle)
    teeth = close_s.rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMA (fastest)
    lips = close_s.rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 8, 5, 50)  # volume MA, Alligator SMAs, and EMA50 alignment warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (lips crosses below teeth) OR price crosses below 8-bar SMA
            if curr_lips <= curr_teeth or curr_close < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (lips crosses above teeth) OR price crosses above 8-bar SMA
            if curr_lips >= curr_teeth or curr_close > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Alligator bullish alignment: lips > teeth > jaw
            # AND volume confirmation AND price > 1d EMA50
            if curr_lips > curr_teeth and curr_teeth > curr_jaw and vol_conf and curr_close > curr_ema50:
                signals[i] = 0.25
                position = 1
            # Short when Alligator bearish alignment: lips < teeth < jaw
            # AND volume confirmation AND price < 1d EMA50
            elif curr_lips < curr_teeth and curr_teeth < curr_jaw and vol_conf and curr_close < curr_ema50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals