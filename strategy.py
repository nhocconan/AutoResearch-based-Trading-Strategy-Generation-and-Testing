#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams Alligator + 1w EMA200 trend filter + volume confirmation
    # Long: price > Alligator Jaw (13) + price > Alligator Teeth (8) + price > Alligator Lips (5) + price > 1w EMA200 + volume > 1.5x 20-period average
    # Short: price < Alligator Jaw + price < Alligator Teeth + price < Alligator Lips + price < 1w EMA200 + volume > 1.5x 20-period average
    # Exit: price crosses Alligator Jaw or 1w EMA200
    # Using 1d timeframe for lower trade frequency, Williams Alligator for trend identification,
    # 1w EMA200 for strong multi-week trend filter, and volume confirmation to avoid false signals.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Alligator (Smoothed Moving Average - SMMA)
    # Jaw: 13-period SMMA of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward  
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price_1d = (high_1d + low_1d) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Shift according to Alligator specification
    jaw_1d_shifted = np.full_like(jaw_1d, np.nan)
    teeth_1d_shifted = np.full_like(teeth_1d, np.nan)
    lips_1d_shifted = np.full_like(lips_1d, np.nan)
    
    if len(jaw_1d) > 8:
        jaw_1d_shifted[8:] = jaw_1d[:-8]
    if len(teeth_1d) > 5:
        teeth_1d_shifted[5:] = teeth_1d[:-5]
    if len(lips_1d) > 3:
        lips_1d_shifted[3:] = lips_1d[:-3]
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 with min_periods
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_1w[199] = np.mean(close_1w[:200])  # SMA200 as seed
        multiplier = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align indicators to 1d timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d_shifted)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d_shifted)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d_shifted)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alligator = (close[i] > jaw_1d_aligned[i] and 
                            close[i] > teeth_1d_aligned[i] and 
                            close[i] > lips_1d_aligned[i])
        bearish_alligator = (close[i] < jaw_1d_aligned[i] and 
                            close[i] < teeth_1d_aligned[i] and 
                            close[i] < lips_1d_aligned[i])
        
        # Trend filter from 1w EMA200
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]
        
        # Entry logic: Alligator alignment + trend alignment + volume confirmation
        long_entry = bullish_alligator and bullish_trend and volume_spike[i]
        short_entry = bearish_alligator and bearish_trend and volume_spike[i]
        
        # Exit logic: Alligator misalignment or trend reversal
        long_exit = not bullish_alligator or not bullish_trend
        short_exit = not bearish_alligator or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_williams_alligator_ema200_volume_v1"
timeframe = "1d"
leverage = 1.0