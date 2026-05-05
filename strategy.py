#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3 AND 4h close > 4h EMA50 AND volume > 1.5 * 20-period average volume
# Short when price breaks below 1h Camarilla S3 AND 4h close < 4h EMA50 AND volume > 1.5 * 20-period average volume
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-30 trades/year per symbol (60-120 over 4 years).
# Camarilla provides structure; 4h EMA50 filters trend; volume confirmation ensures conviction.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Session filter (08-20 UTC) reduces noise trades outside active market hours.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (active market hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data ONCE before loop for Camarilla calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels based on previous 1h bar
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla calculation: based on previous bar's range
    range_1h = high_1h - low_1h
    camarilla_h5 = close_1h + (range_1h * 1.1 / 2)  # R3 level
    camarilla_l5 = close_1h - (range_1h * 1.1 / 2)  # S3 level
    
    # Shift to use previous bar's levels (breakout of previous bar's Camarilla)
    camarilla_h5 = np.roll(camarilla_h5, 1)
    camarilla_l5 = np.roll(camarilla_l5, 1)
    camarilla_h5[0] = np.nan  # First value invalid after roll
    camarilla_l5[0] = np.nan
    
    # Align Camarilla levels to prices timeframe (1h)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1h, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1h, camarilla_l5)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_4h = close_4h > ema_50_4h
    downtrend_4h = close_4h < ema_50_4h
    
    # Align 4h trend to 1h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade during active market hours
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Camarilla R3 AND 4h uptrend AND volume confirmation
            if (close[i] > camarilla_h5_aligned[i] and 
                uptrend_4h_aligned[i] > 0.5 and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price < Camarilla S3 AND 4h downtrend AND volume confirmation
            elif (close[i] < camarilla_l5_aligned[i] and 
                  downtrend_4h_aligned[i] > 0.5 and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < Camarilla S3 OR 4h trend changes to downtrend
            if (close[i] < camarilla_l5_aligned[i] or 
                downtrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > Camarilla R3 OR 4h trend changes to uptrend
            if (close[i] > camarilla_h5_aligned[i] or 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals