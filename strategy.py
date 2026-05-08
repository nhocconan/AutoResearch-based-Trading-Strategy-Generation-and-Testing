#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Williams_Alligator_RSI_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Alligator (Williams Alligator: 13,8,5 SMAs)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Alligator lines (SMA of median price)
    median_price_1w = (high_1w + low_1w) / 2
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values  # 13-period SMA, 8 bars ahead
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values    # 8-period SMA, 5 bars ahead
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values     # 5-period SMA, 3 bars ahead
    
    # Align to daily timeframe (Alligator values available after weekly bar close)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Daily RSI(14)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment), RSI > 50, volume spike
            long_cond = (lips_aligned[i] > teeth_aligned[i] and 
                        teeth_aligned[i] > jaw_aligned[i] and
                        rsi[i] > 50 and
                        volume_spike[i])
            
            # Short: Lips < Teeth < Jaw (bearish alignment), RSI < 50, volume spike
            short_cond = (lips_aligned[i] < teeth_aligned[i] and 
                         teeth_aligned[i] < jaw_aligned[i] and
                         rsi[i] < 50 and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator lines cross (Lips < Teeth) or RSI < 40
            if lips_aligned[i] < teeth_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator lines cross (Lips > Teeth) or RSI > 60
            if lips_aligned[i] > teeth_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals