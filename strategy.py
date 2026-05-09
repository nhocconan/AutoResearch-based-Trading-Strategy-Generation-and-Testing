#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with 1w EMA50 trend filter and volume confirmation.
# The Alligator identifies trend presence and direction via SMAs (Jaw=13, Teeth=8, Lips=5).
# In trending markets, Lips > Teeth > Jaw (uptrend) or reverse (downtrend).
# Combined with 1w EMA50 for higher timeframe trend alignment and volume spike (>1.5x average)
# for confirmation. Designed to capture strong trends in both bull and bear markets
# while avoiding false signals in ranging conditions.
# Target: 30-100 total trades over 4 years (7-25/year).
name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator SMAs (13, 8, 5 periods)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values    # Teeth (8)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values     # Lips (5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need 13 periods for Jaw (slowest SMA)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_50_1w_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 (uptrend) AND volume > 1.5x average
            if lips_val > teeth_val and teeth_val > jaw_val and close[i] > ema_1w and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Jaw > Teeth > Lips (bearish alignment) AND price < 1w EMA50 (downtrend) AND volume > 1.5x average
            elif jaw_val > teeth_val and teeth_val > lips_val and close[i] < ema_1w and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish alignment forms OR trend reverses (price < 1w EMA50)
            if jaw_val > teeth_val or teeth_val > lips_val or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish alignment forms OR trend reverses (price > 1w EMA50)
            if lips_val > teeth_val or teeth_val > jaw_val or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals