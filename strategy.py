#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d ADX Trend + Volume Spike + ATR Stop
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Filter by 1d ADX>25 to ensure we only trade in trending markets (works in both bull and bear). Volume spike confirms institutional participation. ATR stop manages risk. Target: 25-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    def rolling_max(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_20 = rolling_max(high_4h, 20)
    lower_20 = rolling_min(low_4h, 20)
    
    # Align with 1-bar delay (wait for 4h bar close)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    def smoothed_avg(arr, period):
        result = np.full(len(arr), np.nan)
        if len(arr) < period:
            return result
        result[period - 1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result
    
    plus_dm_smooth = smoothed_avg(plus_dm, 14)
    minus_dm_smooth = smoothed_avg(minus_dm, 14)
    tr_smooth = smoothed_avg(tr, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smoothed_avg(dx, 14)
    
    # Align ADX with 1-bar delay
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate ATR(14) for 4h stoploss
    if len(close) >= 14:
        tr1_4h = pd.Series(high).diff().abs()
        tr2_4h = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3_4h = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
        atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    else:
        atr_4h = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        adx_val = adx_aligned[i]
        atr_val = atr_4h[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > upper
        bearish_breakout = curr_close < lower
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: Bullish breakout AND strong trend AND volume spike
            long_condition = bullish_breakout and strong_trend and volume_spike
            # Short: Bearish breakout AND strong trend AND volume spike
            short_condition = bearish_breakout and strong_trend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or break below lower Donchian
            if curr_close <= entry_price - 2.5 * atr_val or curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or break above upper Donchian
            if curr_close >= entry_price + 2.5 * atr_val or curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0