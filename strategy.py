#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Long when price > Alligator Jaw (blue line) AND Teeth (red) > Lips (green) AND price > 1w EMA50 AND volume > 1.5x 20-period average
# Short when price < Alligator Jaw AND Teeth < Lips AND price < 1w EMA50 AND volume > 1.5x 20-period average
# Exit when Alligator lines cross (Teeth == Lips) OR price crosses 1w EMA50
# Williams Alligator identifies trend strength via smoothed median prices (Jaw=13, Teeth=8, Lips=5)
# 1w EMA50 provides long-term trend filter effective in both bull and bear markets
# Volume confirmation reduces false signals
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Timeframe: 1d (primary), HTF: 1w

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike_1.5x"
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
    
    # Get 1d data ONCE before loop for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Alligator (Smoothed Median Price)
    # Median price = (high + low) / 2
    median_price = (high_1d + low_1d) / 2.0
    
    # Jaw (Blue line): 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (Red line): 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (Green line): 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation on 1d (threshold: 1.5x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Jaw AND Teeth > Lips AND price > EMA50 (uptrend) AND volume spike
            if (close[i] > jaw_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND Teeth < Lips AND price < EMA50 (downtrend) AND volume spike
            elif (close[i] < jaw_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Teeth crosses below Lips (Trend weakening) OR price < EMA50
            if teeth_aligned[i] < lips_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Teeth crosses above Lips (Trend weakening) OR price > EMA50
            if teeth_aligned[i] > lips_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals