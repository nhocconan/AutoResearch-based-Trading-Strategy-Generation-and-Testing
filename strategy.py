#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams Alligator with 1w EMA21 trend filter and volume confirmation (>1.5x average)
    # Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets
    # 1w EMA21 filters for primary trend alignment to avoid counter-trend whipsaws
    # Volume spike >1.5x 20-period average confirms institutional participation
    # Target: 15-25 trades/year (60-100 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Get 1w EMA21 for trend filter
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Get 1d volume for confirmation (>1.5x 20-period average)
    vol_ma_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_1d[i] = np.mean(volume[i-20:i])
    volume_spike_1d = volume > (1.5 * vol_ma_1d)
    
    # Align all indicators to LTF (1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(volume_spike_1d[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # 1w trend filter (EMA21)
        bullish_trend = close[i] > ema21_1w_aligned[i]
        bearish_trend = close[i] < ema21_1w_aligned[i]
        
        # Entry logic: Alligator alignment + trend alignment + volume confirmation
        long_entry = bullish_alligator and bullish_trend and volume_spike_1d[i]
        short_entry = bearish_alligator and bearish_trend and volume_spike_1d[i]
        
        # Exit logic: Alligator reversal or trend reversal
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

name = "1d_1w_williams_alligator_ema21_volume_v1"
timeframe = "1d"
leverage = 1.0