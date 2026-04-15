#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w Trend Filter + Volume Confirmation
# Uses Williams Alligator (SMAs with forward shift) to identify trends on daily timeframe,
# confirmed by 1-week EMA trend direction and volume spike.
# Only takes long when price > Alligator Jaw and weekly trend up, short when price < Alligator Jaw and weekly trend down.
# Aims for 30-100 total trades over 4 years with disciplined entries in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 1d (13,8,5 SMAs with future shift)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 1-week EMA for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Alligator condition: lips > jaws = bullish, lips < jaws = bearish
        bullish_alligator = lips_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < jaw_aligned[i]
        
        # Weekly trend: price above/below weekly EMA
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Long entry: bullish alligator + weekly uptrend + volume spike
        if bullish_alligator and weekly_uptrend and volume_spike and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alligator + weekly downtrend + volume spike
        elif bearish_alligator and weekly_downtrend and volume_spike and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: alligator reverses or volume dries up
        elif position == 1 and (not bullish_alligator or not volume_spike):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_alligator or not volume_spike):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0