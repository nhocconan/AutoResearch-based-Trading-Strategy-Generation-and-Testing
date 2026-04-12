#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
    # Alligator (Jaw=TEETH=LIPS) identifies trend absence/presence; trade only when all three aligned
    # Uses 1d EMA50 for higher-timeframe trend filter to avoid counter-trend trades
    # Volume spike (>1.8x 20-period average) confirms breakout strength
    # Designed for low trade frequency (target: 20-40/year) to minimize fee drag
    # Works in bull/bear markets by only trading strong aligned trends
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_4h, 13)  # Blue line
    teeth = smma(close_4h, 8)  # Red line
    lips = smma(close_4h, 5)   # Green line
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h volume for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = np.full(len(df_4h), np.nan)
    for i in range(20, len(df_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i-20:i])
    
    # Volume confirmation: volume > 1.8 * 20-period average (4h)
    volume_spike_4h = volume_4h > (1.8 * vol_ma_4h)
    
    # Align all indicators to LTF (15m)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alligator = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        # 1d trend filter
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Alligator alignment + trend filter + volume confirmation
        long_entry = False
        short_entry = False
        
        # Long: bullish Alligator alignment + bullish 1d trend + volume spike
        if bullish_alligator and bullish_trend:
            long_entry = volume_spike_aligned[i]
        # Short: bearish Alligator alignment + bearish 1d trend + volume spike
        elif bearish_alligator and bearish_trend:
            short_entry = volume_spike_aligned[i]
        
        # Exit logic: Alligator sleeping (no clear trend) or trend reversal
        # Sleeping: jaws intertwined (no clear separation)
        jaw_teeth_close = abs(jaw_aligned[i] - teeth_aligned[i]) < (close[i] * 0.001)
        teeth_lips_close = abs(teeth_aligned[i] - lips_aligned[i]) < (close[i] * 0.001)
        alligator_sleeping = jaw_teeth_close and teeth_lips_close
        
        long_exit = bearish_trend or alligator_sleeping
        short_exit = bullish_trend or alligator_sleeping
        
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

name = "4h_1d_alligator_trend_volume_v1"
timeframe = "4h"
leverage = 1.0