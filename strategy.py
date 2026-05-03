#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with 1w EMA trend filter and volume confirmation.
# Long: Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA50 AND volume > 1.5x 20-period MA
# Short: Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA50 AND volume > 1.5x 20-period MA
# Exit: Opposite Alligator alignment OR 1w EMA trend fails OR volume drops.
# Uses Williams Alligator for trend identification, 1w EMA for higher-timeframe filter, volume for confirmation.
# Discrete sizing 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# Alligator avoids whipsaws in ranging markets, 1w EMA filters for strong trends, volume reduces false signals.

name = "1d_WilliamsAlligator_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (13/8/5 SMAs)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Alligator and 1w EMA50 to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume regime: current 1d volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine Alligator alignment
        is_bullish = lips_val > teeth_val > jaw_val  # Lips > Teeth > Jaw
        is_bearish = lips_val < teeth_val < jaw_val  # Lips < Teeth < Jaw
        
        # Entry logic
        if position == 0:
            # Long: Bullish Alligator AND price above 1w EMA50 AND volume spike
            if is_bullish and close_val > ema_50_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND price below 1w EMA50 AND volume spike
            elif is_bearish and close_val < ema_50_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator OR price below 1w EMA50 OR volume drops
            if is_bearish or close_val < ema_50_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator OR price above 1w EMA50 OR volume drops
            if is_bullish or close_val > ema_50_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals