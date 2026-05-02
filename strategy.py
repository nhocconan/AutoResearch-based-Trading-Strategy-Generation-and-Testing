#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume spike confirmation
# Uses 6h timeframe for Williams Alligator (jaw/teeth/lips) to identify trend and entry
# 1d EMA34 as trend filter ensures alignment with daily trend
# Volume confirmation (2.0x 24-period average) ensures institutional participation
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - long when Lips > Teeth > Jaw, short when reverse
# Works in bull markets via trend-following entries, in bear via same logic (short signals)
# Discrete position sizing (0.25) to minimize fee churn while maintaining adequate exposure

name = "6h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h data
    # Jaw: 13-period SMMA, 8 periods ahead
    # Teeth: 8-period SMMA, 5 periods ahead  
    # Lips: 5-period SMMA, 3 periods ahead
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw.shift(8)  # 8 periods ahead
    
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth.shift(5)  # 5 periods ahead
    
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips.shift(3)  # 3 periods ahead
    
    # Volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(lips.iloc[i]) or 
            np.isnan(teeth.iloc[i]) or np.isnan(jaw.iloc[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Get current Alligator values
        lips_val = lips.iloc[i]
        teeth_val = teeth.iloc[i]
        jaw_val = jaw.iloc[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA34 + volume confirm
            if lips_val > teeth_val and teeth_val > jaw_val and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA34 + volume confirm
            elif lips_val < teeth_val and teeth_val < jaw_val and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Lips < Teeth (bullish alignment broken) or price < 1d EMA34
            if lips_val < teeth_val or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Lips > Teeth (bearish alignment broken) or price > 1d EMA34
            if lips_val > teeth_val or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals