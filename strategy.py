#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Alligator (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get daily data for Elder Ray (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate Williams Alligator on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Smoothed moving averages (Jaw: 13, Teeth: 8, Lips: 5)
    def smoothed_ma(data, period):
        sma = np.full(len(data), np.nan)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smoothed_ma(close_1w, 13)
    teeth = smoothed_ma(close_1w, 8)
    lips = smoothed_ma(close_1w, 5)
    
    # Calculate Elder Ray on daily (13-period EMA)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13
    ema13 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        ema13[12] = np.mean(close_1d[:13])
        for i in range(13, len(close_1d)):
            ema13[i] = (close_1d[i] * 2 + ema13[i-1] * 12) / 14
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter (20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup period
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator condition: Lips > Teeth > Jaw = bullish alignment
        alligator_bull = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_bear = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray condition: Bull Power > 0 and Bear Power < 0 for strong trend
        elder_bull = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        elder_bear = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
        
        # Volume confirmation: > 1.5x average volume
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Long: Alligator bullish + Elder Ray bull + volume
            if alligator_bull and elder_bull and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Elder Ray bear + volume
            elif alligator_bear and elder_bear and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish OR Elder Ray turns bearish
            if not alligator_bull or not elder_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Elder Ray turns bullish
            if not alligator_bear or not elder_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1W_1D_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0