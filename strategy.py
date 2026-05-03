#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams Alligator + volume confirmation + 4h/1d EMA trend filter + session filter (08-20 UTC)
# Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period)
# Long when Lips > Teeth > Jaw (bullish alignment) + price > Jaw + volume spike + above 4h/1d EMA(50)
# Short when Lips < Teeth < Jaw (bearish alignment) + price < Jaw + volume spike + below 4h/1d EMA(50)
# Uses 4h and 1d EMA(50) for stronger trend alignment to reduce whipsaw in choppy markets
# Session filter reduces noise trades during low-volume hours
# Designed for low trade frequency (15-37/year) to minimize fee drag. Works in both bull and bear markets.

name = "1h_WilliamsAlligator_Volume_4h1dEMA50_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h and 1d data for EMA(50) trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h and 1d for trend filters
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h and 1d EMA to 1h timeframe (wait for completed HTF bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components on 1h
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period smoothed median (SMMA)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    
    # Teeth: 8-period smoothed median (SMMA)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    
    # Lips: 5-period smoothed median (SMMA)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation (2.0x 20-period average) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 60  # max(13,8,5 for Alligator + 50 for 4h/1d EMA + 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long entry: Bullish Alligator + price > Jaw + above 4h/1d EMA(50) + volume spike + in session
            if (bullish_alligator and close[i] > jaw[i] and 
                close[i] > ema_50_4h_aligned[i] and close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: Bearish Alligator + price < Jaw + below 4h/1d EMA(50) + volume spike + in session
            elif (bearish_alligator and close[i] < jaw[i] and 
                  close[i] < ema_50_4h_aligned[i] and close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment OR price below Jaw OR below 4h/1d EMA(50) OR below 1d EMA(50)
            bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if (bearish_alligator or close[i] < jaw[i] or 
                close[i] < ema_50_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment OR price above Jaw OR above 4h/1d EMA(50) OR above 1d EMA(50)
            bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if (bullish_alligator or close[i] > jaw[i] or 
                close[i] > ema_50_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals