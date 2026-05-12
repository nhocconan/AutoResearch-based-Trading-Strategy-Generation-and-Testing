#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
# Hypothesis: Use Camarilla pivot levels on 12h for precise entry/exit, filtered by 1d EMA trend and volume confirmation.
# Long when price breaks above R3 with volume spike and 1d EMA up, short when breaks below S3 with volume spike and 1d EMA down.
# Designed for low frequency (12-37 trades/year) to avoid fee drift. Works in both bull and bear via trend filter.
# Uses Camarilla levels (R3/S3) as strong support/resistance, with volume confirmation to avoid false breakouts.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close arrays.
    Returns R3, R2, R1, PP, S1, S2, S3 arrays.
    """
    typical = (high + low + close) / 3
    range_ = high - low
    
    R3 = close + (range_ * 1.1 / 2)
    R2 = close + (range_ * 1.1 / 4)
    R1 = close + (range_ * 1.1 / 6)
    PP = typical
    S1 = close - (range_ * 1.1 / 6)
    S2 = close - (range_ * 1.1 / 4)
    S3 = close - (range_ * 1.1 / 2)
    
    return R3, R2, R1, PP, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels
    R3, R2, R1, PP, S1, S2, S3 = calculate_camarilla(high, low, close)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA to be stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price relative to 1d EMA
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike and 1d EMA up
            if close[i] > R3[i] and volume_spike and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and 1d EMA down
            elif close[i] < S3[i] and volume_spike and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below PP or opposite signal with volume
            if close[i] < PP[i] or (close[i] < S3[i] and volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above PP or opposite signal with volume
            if close[i] > PP[i] or (close[i] > R3[i] and volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals