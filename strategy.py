#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator (13,8,5 SMAs) identifies trends via jaw-teeth-lips alignment.
# Long when lips > teeth > jaw (bullish alignment), short when lips < teeth < jaw (bearish).
# Combined with 1d EMA50 trend filter and volume spikes to avoid whipsaws.
# Works in both bull and bear markets by taking signals only in direction of higher timeframe trend.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier50 = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier50 + ema50_1d[i-1]
    
    # Calculate Williams Alligator components (13,8,5 SMAs)
    # Jaw: 13-period SMMA (smoothed moving average)
    jaw = np.full(n, np.nan)
    for i in range(12, n):
        jaw[i] = np.mean(close[i-12:i+1])
    
    # Teeth: 8-period SMMA
    teeth = np.full(n, np.nan)
    for i in range(7, n):
        teeth[i] = np.mean(close[i-7:i+1])
    
    # Lips: 5-period SMMA
    lips = np.full(n, np.nan)
    for i in range(4, n):
        lips[i] = np.mean(close[i-4:i+1])
    
    # Align daily EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate average volume (24-period = 12 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        
        # Williams Alligator values
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Bullish alignment (lips > teeth > jaw) + above daily EMA50 + volume confirmation
            if (lips_val > teeth_val and teeth_val > jaw_val and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bearish alignment (lips < teeth < jaw) + below daily EMA50 + volume confirmation
            elif (lips_val < teeth_val and teeth_val < jaw_val and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alignment breaks down (lips <= teeth) or trend turns down
            if (lips_val <= teeth_val or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alignment breaks down (lips >= teeth) or trend turns up
            if (lips_val >= teeth_val or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_WilliamsAlligator_Trend_Volume"
timeframe = "12h"
leverage = 1.0