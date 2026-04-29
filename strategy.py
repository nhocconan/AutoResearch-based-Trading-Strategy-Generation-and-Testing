#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Williams Alligator identifies trend via Jaw (13), Teeth (8), Lips (5) SMAs.
# Long when Lips > Teeth > Jaw and price > 1d EMA50; Short when Lips < Teeth < Jaw and price < 1d EMA50.
# Volume confirmation ensures institutional participation; discrete sizing (0.25) minimizes fee churn.
# Works in both bull/bear markets: Alligator catches sustained moves, EMA50 filters counter-trend noise.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_Williams_Alligator_1dEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components on 12h timeframe
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = close_s.rolling(window=8, min_periods=8).mean().values   # Teeth (8)
    lips = close_s.rolling(window=5, min_periods=5).mean().values    # Lips (5)
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 8, 5, 20)  # 1d EMA50, Alligator components, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Alligator alignment
        bullish_alignment = curr_lips > curr_teeth and curr_teeth > curr_jaw
        bearish_alignment = curr_lips < curr_teeth and curr_teeth < curr_jaw
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator loses bullish alignment OR price crosses below 1d EMA50
            if not bullish_alignment or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator loses bearish alignment OR price crosses above 1d EMA50
            if not bearish_alignment or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish Alligator alignment + price above 1d EMA50 + volume confirmation
            if bullish_alignment and curr_close > curr_ema_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment + price below 1d EMA50 + volume confirmation
            elif bearish_alignment and curr_close < curr_ema_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals