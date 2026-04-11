#!/usr/bin/env python3
# 6h_1d_williams_alligator_v1
# Strategy: 6s Williams Alligator with 1d trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (sleeping) vs presence (awake).
# In ranging markets, lines intertwine (sleep) → no trade. When lines diverge in order (Lips > Teeth > Jaw for uptrend, reverse for downtrend) with 1d trend alignment and volume confirmation → trend is awake and tradable.
# Works in bull/bear by only trading when trend is clear, avoiding whipsaws in ranges. Low frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    # Jaw: Blue line - 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: Red line - 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: Green line - 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after max shift (8) + jaw period (13) - 1
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw.iloc[i]) or 
            np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Alligator alignment: check if lines are properly ordered and separated
        jaw_val = jaw.iloc[i]
        teeth_val = teeth.iloc[i]
        lips_val = lips.iloc[i]
        
        # Awesome oscillator alternative: check for clear separation
        # Uptrend: Lips > Teeth > Jaw (green above red above blue)
        # Downtrend: Lips < Teeth < Jaw (green below red below blue)
        # Add minimum separation to avoid whipsaw when lines are close
        min_sep = 0.0001 * close[i]  # 0.01% of price as minimum separation
        
        awake_uptrend = (lips_val > teeth_val + min_sep) and (teeth_val > jaw_val + min_sep)
        awake_downtrend = (lips_val < teeth_val - min_sep) and (teeth_val < jaw_val - min_sep)
        
        # Entry logic: Alligator awake + 1d trend alignment + volume confirmation
        if awake_uptrend and uptrend_1d and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif awake_downtrend and downtrend_1d and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Alligator goes back to sleep (lines intertwine) or trend fails
        elif position == 1 and (not awake_uptrend or not uptrend_1d):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not awake_downtrend or not downtrend_1d):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals