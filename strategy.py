#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter and volume confirmation
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA50 + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA50 + volume spike
# Exit when Alligator alignment breaks or price crosses 1d EMA50
# Williams Alligator catches trend changes early; EMA50 filters counter-trend moves
# Volume spike confirms institutional participation. Designed for low frequency (15-30/year)
# Works in bull markets via trend following, in bear via short signals on alignment breakdown

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 6h data (Smoothed Median with periods)
    # Jaw: Blue line - 13-period SMMA, shifted 8 bars forward
    # Teeth: Red line - 8-period SMMA, shifted 5 bars forward  
    # Lips: Green line - 5-period SMMA, shifted 3 bars forward
    median = (prices['high'].values + prices['low'].values) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(median, 13)
    teeth_raw = smma(median, 8)
    lips_raw = smma(median, 5)
    
    # Shift forward as per Alligator definition (Jaw +8, Teeth +5, Lips +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: bullish alignment + uptrend + volume spike
            if bullish_alignment and price > ema_50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + downtrend + volume spike
            elif bearish_alignment and price < ema_50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: alignment breaks or price crosses EMA50
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or price < EMA50
                if not bullish_alignment or price < ema_50_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or price > EMA50
                if not bearish_alignment or price > ema_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0