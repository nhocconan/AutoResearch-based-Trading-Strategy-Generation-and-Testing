#!/usr/bin/env python3
"""
1d Williams Alligator + 1w EMA Trend + Volume Spike
Long: Jaw < Teeth < Lips (bullish alignment) + price > 1w EMA200 + volume > 2x 1d volume SMA(20)
Short: Jaw > Teeth > Lips (bearish alignment) + price < 1w EMA200 + volume > 2x 1d volume SMA(20)
Exit: Opposite Alligator alignment or price crosses 1w EMA200
Williams Alligator identifies trend phases; EMA200 filters for long-term trend direction.
Designed to work in both bull and bear markets by requiring alignment with weekly trend.
Target: 30-100 total trades over 4 years (7-25/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(200) for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Williams Alligator (13,8,5 smoothed with 8,5,3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # 1d volume SMA(20) for volume filter
    vol_sma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 200)  # need EMA200 and Alligator components
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_sma_1d[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_1d[i]
        ema_200_val = ema_200_1w_aligned[i]
        
        # Williams Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] < lips[i]  # Jaw < Teeth < Lips
        bearish_alignment = jaw[i] > teeth[i] > lips[i]  # Jaw > Teeth > Lips
        
        if position == 0:
            # Long: Bullish Alligator alignment + price > 1w EMA200 + volume spike
            if bullish_alignment and price > ema_200_val and vol > 2.0 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + price < 1w EMA200 + volume spike
            elif bearish_alignment and price < ema_200_val and vol > 2.0 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish Alligator alignment or price < 1w EMA200
            if bearish_alignment or price < ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish Alligator alignment or price > 1w EMA200
            if bullish_alignment or price > ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA200_VolumeSpike"
timeframe = "1d"
leverage = 1.0