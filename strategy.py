#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) with future shifts.
# In trends, the lines are ordered and separated; in ranges, they intertwine.
# Combined with 1d EMA trend filter and volume spikes, it filters false signals and captures trends.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 4h timeframe
    # Jaw: SMA(13) shifted by 8 bars
    jaw = np.full(n, np.nan)
    for i in range(12, n):
        jaw[i] = np.mean(close[i-12:i+1])
    jaw_shifted = np.full(n, np.nan)
    for i in range(8, n):
        jaw_shifted[i] = jaw[i-8]
    
    # Teeth: SMA(8) shifted by 5 bars
    teeth = np.full(n, np.nan)
    for i in range(7, n):
        teeth[i] = np.mean(close[i-7:i+1])
    teeth_shifted = np.full(n, np.nan)
    for i in range(5, n):
        teeth_shifted[i] = teeth[i-5]
    
    # Lips: SMA(5) shifted by 3 bars
    lips = np.full(n, np.nan)
    for i in range(4, n):
        lips[i] = np.mean(close[i-4:i+1])
    lips_shifted = np.full(n, np.nan)
    for i in range(3, n):
        lips_shifted[i] = lips[i-3]
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        
        # Alligator values
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + above 1d EMA50 + volume confirmation
            if (lips_val > teeth_val and
                teeth_val > jaw_val and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Lips < Teeth < Jaw (bearish alignment) + below 1d EMA50 + volume confirmation
            elif (lips_val < teeth_val and
                  teeth_val < jaw_val and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator lines lose bullish alignment or price breaks below 1d EMA
            if not (lips_val > teeth_val and teeth_val > jaw_val) or price < ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator lines lose bearish alignment or price breaks above 1d EMA
            if not (lips_val < teeth_val and teeth_val < jaw_val) or price > ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsAlligator_Trend_Volume"
timeframe = "4h"
leverage = 1.0