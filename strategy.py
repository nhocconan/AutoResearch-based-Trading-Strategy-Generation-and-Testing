#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using Camarilla pivot levels from 1d timeframe with volume confirmation and 1d ADX trend filter.
Long when price breaks above R3 level in uptrend with volume > 2x average. Short when price breaks below S3 level in downtrend with volume confirmation.
Trades only during strong trends to avoid whipsaw in ranging markets. Position size 0.25 to limit drawdown.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day pivot points (using previous day's OHLC)
    # Camarilla levels: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day based on previous day
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_r4 = np.zeros(len(close_1d))
    camarilla_s4 = np.zeros(len(close_1d))
    
    # For each day, calculate levels based on previous day's data
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_range = prev_high - prev_low
        
        # Camarilla levels
        camarilla_r3[i] = prev_close + (prev_range * 1.1 / 4)
        camarilla_s3[i] = prev_close - (prev_range * 1.1 / 4)
        camarilla_r4[i] = prev_close + (prev_range * 1.1 / 2)
        camarilla_s4[i] = prev_close - (prev_range * 1.1 / 2)
    
    # Calculate ADX(14) on 1d data for trend filter
    period = 14
    
    # True Range
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(len(close_1d))
    minus_dm = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed TR, +DM, -DM (Wilder smoothing)
    tr_period = np.zeros(len(close_1d))
    plus_dm_period = np.zeros(len(close_1d))
    minus_dm_period = np.zeros(len(close_1d))
    
    if len(close_1d) >= period:
        tr_period[period-1] = np.sum(tr_1d[1:period+1])
        plus_dm_period[period-1] = np.sum(plus_dm[1:period+1])
        minus_dm_period[period-1] = np.sum(minus_dm[1:period+1])
        
        for i in range(period, len(close_1d)):
            tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr_1d[i]
            plus_dm_period[i] = plus_dm_period[i-1] - (plus_dm_period[i-1] / period) + plus_dm[i]
            minus_dm_period[i] = minus_dm_period[i-1] - (minus_dm_period[i-1] / period) + minus_dm[i]
    
    # Directional Indicators
    plus_di = np.zeros(len(close_1d))
    minus_di = np.zeros(len(close_1d))
    dx = np.zeros(len(close_1d))
    
    for i in range(period-1, len(close_1d)):
        if tr_period[i] > 0:
            plus_di[i] = 100 * (plus_dm_period[i] / tr_period[i])
            minus_di[i] = 100 * (minus_dm_period[i] / tr_period[i])
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX = smoothed DX
    adx_1d = np.zeros(len(close_1d))
    if len(close_1d) >= 2 * period - 1:
        adx_1d[2*period-2] = np.sum(dx[period-1:2*period-1]) / period
        for i in range(2*period-1, len(close_1d)):
            adx_1d[i] = (adx_1d[i-1] * (period - 1) + dx[i]) / period
    
    # Align 1d data to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation on 12h timeframe
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need ADX (28), camarilla levels (1), volume MA (20)
    start_idx = max(2*period-1, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: price breaks above R3 level in uptrend with volume
            if trending and price > camarilla_r3_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below S3 level in downtrend with volume
            elif not trending and price < camarilla_s3_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below S3 level or trend weakens
            if price < camarilla_s3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above R3 level or trend weakens
            if price > camarilla_r3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_1dADX25_Volume"
timeframe = "12h"
leverage = 1.0