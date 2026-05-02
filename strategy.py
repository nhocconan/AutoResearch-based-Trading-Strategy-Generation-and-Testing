#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence via SMAs
# 1w EMA50 ensures alignment with major trend to avoid counter-trend trades
# Volume confirmation filters false signals. Target: 30-100 total trades over 4 years on 1d timeframe
# Uses discrete position sizing (0.30) to balance return and drawdown control
# Works in bull markets (Lips > Teeth > Jaw + 1w EMA50 up) and bear markets (Lips < Teeth < Jaw + 1w EMA50 down)

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for Williams Alligator (SMAs: Jaw=13, Teeth=8, Lips=5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator calculation using SMAs
    jaw = pd.Series(df_1d['close'].values).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(df_1d['close'].values).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 1d timeframe (wait for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator calculation)
    start_idx = 13
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment (trending market)
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Determine 1w EMA50 trend
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment with volume confirmation and uptrend
            if bullish_alligator and volume_confirmation[i] and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Bearish Alligator alignment with volume confirmation and downtrend
            elif bearish_alligator and volume_confirmation[i] and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator loses bullish alignment OR trend changes to downtrend
            if not bullish_alligator or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Alligator loses bearish alignment OR trend changes to uptrend
            if not bearish_alligator or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals