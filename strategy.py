#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 1w close > 1w EMA34 AND volume > 1.5 * 20-period avg volume
# Short when price breaks below Camarilla S3 AND 1w close < 1w EMA34 AND volume > 1.5 * 20-period avg volume
# Uses discrete sizing (0.30) to limit fee drag. Target: 12-37 trades/year per symbol.
# Camarilla provides structure; 1w EMA34 filters trend to avoid counter-trend trades; volume confirms conviction.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# 12h timeframe minimizes fee drag while capturing multi-day moves.

name = "12h_Camarilla_R3S3_1wEMA34_VolumeConfirm"
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
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3) based on previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_high = close_12h + (high_12h - low_12h) * 1.1 / 4
    camarilla_low = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    # Shift to use previous bar's levels (breakout of previous bar's Camarilla)
    camarilla_high = np.roll(camarilla_high, 1)
    camarilla_low = np.roll(camarilla_low, 1)
    camarilla_high[0] = np.nan  # First value invalid after roll
    camarilla_low[0] = np.nan
    
    # Align Camarilla levels to prices timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align 1w trend to prices timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Camarilla R3 AND 1w uptrend AND volume confirmation
            if (close[i] > camarilla_high_aligned[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price < Camarilla S3 AND 1w downtrend AND volume confirmation
            elif (close[i] < camarilla_low_aligned[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price < Camarilla S3 OR 1w trend changes to downtrend
            if (close[i] < camarilla_low_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price > Camarilla R3 OR 1w trend changes to uptrend
            if (close[i] > camarilla_high_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals