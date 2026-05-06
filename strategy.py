#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend + volume spike confirmation
# Long when 12h jaw < teeth < lips (bullish alignment) AND 1d close > 1d EMA50 AND volume > 2.0 * 20-bar avg volume
# Short when 12h jaw > teeth > lips (bearish alignment) AND 1d close < 1d EMA50 AND volume > 2.0 * 20-bar avg volume
# Exit when Alligator alignment breaks (jaws cross teeth or lips) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies strong trends via smoothed medians (Jaw=13, Teeth=8, Lips=5)
# 1d EMA50 filters for higher timeframe trend alignment (proven BTC/ETH edge)
# Volume spike confirmation reduces false signals during low participation
# Works in both bull and bear markets by following the 1d trend and Alligator alignment

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
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
    
    # Calculate 12h Williams Alligator and 1d EMA50 ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 13 or len(df_1d) < 50:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator lines (smoothed medians)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_12h, 13)
    teeth = smma(median_12h, 8)
    lips = smma(median_12h, 5)
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed bars)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish Alligator alignment (jaw < teeth < lips) AND uptrend AND volume spike
            if jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment (jaw > teeth > lips) AND downtrend AND volume spike
            elif jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks bearish OR volume drops below average
            if not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks bullish OR volume drops below average
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals