#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w Trend + Volume Spike
# Williams Alligator from 1d timeframe identifies trend via three smoothed SMAs (Jaw, Teeth, Lips).
# Long when Lips > Teeth > Jaw with volume confirmation and 1w uptrend.
# Short when Lips < Teeth < Jaw with volume confirmation and 1w downtrend.
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year).
# Uses discrete position sizing (0.30) to minimize fee churn and manage drawdown.

name = "1d_Williams_Alligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator (1d timeframe)
    # Jaw: 13-period SMMA, 8 bars offset
    # Teeth: 8-period SMMA, 5 bars offset
    # Lips: 5-period SMMA, 3 bars offset
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Align Williams Alligator to 1d timeframe (already aligned as primary)
    jaw_aligned = jaw  # No alignment needed for primary timeframe
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for Alligator
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: Lips > Teeth > Jaw + volume + 1w uptrend
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: Lips < Teeth < Jaw + volume + 1w downtrend
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Lips < Teeth OR 1w trend turns down
            if (lips_aligned[i] < teeth_aligned[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Lips > Teeth OR 1w trend turns up
            if (lips_aligned[i] > teeth_aligned[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals