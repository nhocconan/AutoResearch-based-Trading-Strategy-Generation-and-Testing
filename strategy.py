#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends
# When all three lines are aligned (Jaw < Teeth < Lips for uptrend, Jaw > Teeth > Lips for downtrend),
# it indicates a strong trend. Combined with 1d EMA50 for higher timeframe alignment and volume confirmation
# (>2.0x 20-period average) to filter weak moves. Designed to capture sustained trends in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
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
    
    # Williams Alligator components (smoothed moving averages)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA calculation using EMA as approximation (standard practice)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 8, 5, 20)  # 1d EMA50, Alligator components, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict threshold for fewer trades)
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Alligator alignment conditions
        # Uptrend: Jaw < Teeth < Lips (all lines aligned upward)
        # Downtrend: Jaw > Teeth > Lips (all lines aligned downward)
        alligator_long = curr_jaw < curr_teeth and curr_teeth < curr_lips
        alligator_short = curr_jaw > curr_teeth and curr_teeth > curr_lips
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks OR price closes below 1d EMA50
            if not alligator_long or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR price closes above 1d EMA50
            if not alligator_short or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Alligator uptrend alignment + price above 1d EMA50 + volume confirmation
            if (alligator_long and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator downtrend alignment + price below 1d EMA50 + volume confirmation
            elif (alligator_short and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals