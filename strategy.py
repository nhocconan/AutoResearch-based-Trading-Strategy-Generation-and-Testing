#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend identification and entry timing
# 1w EMA50 ensures we only trade in the direction of the weekly trend to avoid counter-trend whipsaws
# Volume confirmation (2.0x average) filters low-participation breakouts
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years)
# Williams Alligator works in both bull and bear markets by adapting to changing trends
# Prioritizes BTC/ETH performance with SOL as secondary

name = "12h_WilliamsAlligator_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price_12h = (high_12h := (df_12h['high'].values + df_12h['low'].values) / 2)
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values    # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator signals with 1w trend filter
        # Alligator is asleep when Jaw, Teeth, Lips are intertwined (no trend)
        # Alligator awakens when lines separate in a specific order
        # Long: Lips > Teeth > Jaw (green > red > blue) + volume spike + price above 1w EMA50 (uptrend)
        # Short: Lips < Teeth < Jaw (green < red < blue) + volume spike + price below 1w EMA50 (downtrend)
        if position == 0:
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines re-intertwine (Lips crosses below Teeth) OR price below 1w EMA50 (trend change)
            if lips_aligned[i] < teeth_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines re-intertwine (Lips crosses above Teeth) OR price above 1w EMA50 (trend change)
            if lips_aligned[i] > teeth_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals