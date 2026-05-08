#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WilliamsAlligator_1wTrend_Volume"
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
    
    # Get weekly data for Williams Alligator (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams Alligator on weekly data
    median_price_1w = (high_1w + low_1w) / 2
    
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    jaw_1w = pd.Series(median_price_1w).ewm(alpha=1/13, adjust=False).mean().values
    jaw_1w = np.roll(jaw_1w, 8)
    jaw_1w[:8] = np.nan
    
    # Teeth (Red): 8-period SMMA, shifted 5 bars
    teeth_1w = pd.Series(median_price_1w).ewm(alpha=1/8, adjust=False).mean().values
    teeth_1w = np.roll(teeth_1w, 5)
    teeth_1w[:5] = np.nan
    
    # Lips (Green): 5-period SMMA, shifted 3 bars
    lips_1w = pd.Series(median_price_1w).ewm(alpha=1/5, adjust=False).mean().values
    lips_1w = np.roll(lips_1w, 3)
    lips_1w[:3] = np.nan
    
    # Align Alligator lines to daily timeframe
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Daily trend filter: EMA(34)
    ema_34_1d = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or np.isnan(lips_1w_aligned[i]) or 
            np.isnan(ema_34_1d[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > EMA34 + volume
            if (lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i] and
                close[i] > ema_34_1d[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) + price < EMA34 + volume
            elif (jaw_1w_aligned[i] > teeth_1w_aligned[i] > lips_1w_aligned[i] and
                  close[i] < ema_34_1d[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish alignment or price below EMA34
            if (jaw_1w_aligned[i] > teeth_1w_aligned[i] > lips_1w_aligned[i] or
                close[i] < ema_34_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alignment or price above EMA34
            if (lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i] or
                close[i] > ema_34_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals