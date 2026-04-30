#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Williams Alligator uses three SMAs (jaw=13, teeth=8, lips=5) to identify trends.
# Long when lips > teeth > jaw (bullish alignment) with 1d uptrend (close > 1d EMA50) and volume > 1.8x 20-bar avg.
# Short when lips < teeth < jaw (bearish alignment) with 1d downtrend (close < 1d EMA50) and volume > 1.8x 20-bar avg.
# Exit when Alligator alignment breaks (lips crosses teeth or jaw).
# Williams Alligator is effective in both trending and ranging markets, and the 1d EMA50 filter reduces false signals.
# Timeframe: 12h, HTF: 1d as per experiment guidelines.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: jaw (13), teeth (8), lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Alligator
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment (lips > teeth > jaw), uptrend, volume spike
            if (curr_lips > curr_teeth > curr_jaw and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment (lips < teeth < jaw), downtrend, volume spike
            elif (curr_lips < curr_teeth < curr_jaw and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Alligator alignment breaks (lips <= teeth or teeth <= jaw)
            if curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Alligator alignment breaks (lips >= teeth or teeth >= jaw)
            if curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals