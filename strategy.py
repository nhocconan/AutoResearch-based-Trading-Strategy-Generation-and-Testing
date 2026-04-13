#!/usr/bin/env python3
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
    
    # Get daily data for trend and volatility calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily
    high_20 = np.full(len(close_1d), np.nan)
    low_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        high_20[i] = np.max(high_1d[i-20:i])
        low_20[i] = np.min(low_1d[i-20:i])
    
    # Calculate 14-period ATR on daily for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Get weekly data for regime filter (choppiness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for Choppiness Index
    tr1_w = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2_w = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1_w)
    tr_w = np.concatenate([[np.nan], tr2_w])
    
    # Calculate ADX-like components for chop filter
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    # Smooth the DM and TR for DI calculation
    def wilders_smooth(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr_14_w = wilders_smooth(tr_w, 14)
    plus_di_14_w = 100 * wilders_smooth(plus_dm, 14) / tr_14_w
    minus_di_14_w = 100 * wilders_smooth(minus_dm, 14) / tr_14_w
    dx_14_w = 100 * np.abs(plus_di_14_w - minus_di_14_w) / (plus_di_14_w + minus_di_14_w)
    adx_14_w = wilders_smooth(dx_14_w, 14)
    
    # Choppiness Index approximation: when ADX is low, market is choppy
    chop_threshold = 25  # ADX < 25 indicates choppy/ranging market
    
    # Align indicators to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    adx_14_w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_w, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or
            np.isnan(adx_14_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is reasonable
        vol_filter = atr_14_aligned[i] > 0
        
        # Regime filter: avoid trading in strong trends (ADX > 25), favor choppy markets
        regime_filter = adx_14_w_aligned[i] < chop_threshold
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        # Entry conditions: breakout in choppy market with volume confirmation
        long_entry = long_breakout and vol_filter and regime_filter
        short_entry = short_breakout and vol_filter and regime_filter
        
        # Exit conditions: opposite breakout or volatility expansion
        exit_long = position == 1 and (short_breakout or not vol_filter)
        exit_short = position == -1 and (long_breakout or not vol_filter)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_chop_breakout"
timeframe = "1d"
leverage = 1.0