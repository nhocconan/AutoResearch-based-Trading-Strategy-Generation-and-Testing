#!/usr/bin/env python3
"""
6h Elder Ray Power + ADX Trend + Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure relative to trend.
Combined with ADX (>25) for trend strength and volume confirmation, this captures strong directional moves in both bull and bear markets.
Uses discrete sizing (0.25) to manage drawdowns. Targets 50-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA13 (Elder Ray) and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # 1d ADX calculation
    def calculate_adx(high_arr, low_arr, close_arr, window=14):
        # True Range
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Directional Movement
        up_move = high_arr[1:] - high_arr[:-1]
        down_move = low_arr[:-1] - low_arr[1:]
        up_move = np.concatenate([[0], up_move])
        down_move = np.concatenate([[0], down_move])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        return adx
    
    adx_values = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13 warmup, ADX, and volume MA
    start_idx = max(50, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Elder Ray Power
        bull_power = curr_high - ema_13_aligned[i]  # High - EMA13
        bear_power = ema_13_aligned[i] - curr_low   # EMA13 - Low
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Look for entry signals - require: Elder Ray power + trend + volume
            # Long: bullish power positive AND strong trend AND volume spike
            long_entry = (bull_power > 0) and strong_trend and vol_spike
            # Short: bearish power positive AND strong trend AND volume spike
            short_entry = (bear_power > 0) and strong_trend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: bull power turns negative OR loss of trend OR no volume confirmation
            if (bull_power <= 0) or (adx_aligned[i] <= 20) or (not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: bear power turns negative OR loss of trend OR no volume confirmation
            if (bear_power <= 0) or (adx_aligned[i] <= 20) or (not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_ADX_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0