#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly ADX trend filter + Daily Donchian breakout with volume confirmation.
# Weekly ADX > 25 identifies strong trends (works in both bull and bear markets).
# Daily Donchian(20) breakout captures momentum in trending markets.
# Volume confirmation ensures breakouts have conviction.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        tr_1w[i] = max(high_1w[i] - low_1w[i], 
                       abs(high_1w[i] - close_1w[i-1]), 
                       abs(low_1w[i] - close_1w[i-1]))
    
    # Directional Movement
    plus_dm_1w = np.zeros(len(close_1w))
    minus_dm_1w = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        plus_dm_1w[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_1w[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    def smooth(values, period):
        smoothed = np.zeros_like(values)
        for i in range(len(values)):
            if i < period:
                smoothed[i] = np.nan
            elif i == period:
                smoothed[i] = np.sum(values[1:i+1])
            else:
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr_1w = smooth(tr_1w, 14)
    plus_di_1w = 100 * smooth(plus_dm_1w, 14) / atr_1w
    minus_di_1w = 100 * smooth(minus_dm_1w, 14) / atr_1w
    dx_1w = np.zeros(len(close_1w))
    for i in range(len(close_1w)):
        if plus_di_1w[i] + minus_di_1w[i] != 0:
            dx_1w[i] = 100 * abs(plus_di_1w[i] - minus_di_1w[i]) / (plus_di_1w[i] + minus_di_1w[i])
        else:
            dx_1w[i] = 0
    adx_1w = smooth(dx_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    period = 20
    donchian_high_1d = np.full(len(close_1d), np.nan)
    donchian_low_1d = np.full(len(close_1d), np.nan)
    
    for i in range(period-1, len(close_1d)):
        donchian_high_1d[i] = np.max(high_1d[i-period+1:i+1])
        donchian_low_1d[i] = np.min(low_1d[i-period+1:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Calculate average volume (20-period) on daily data
    volume_1d = df_1d['volume'].values
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i-20:i])
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        adx_val = adx_1w_aligned[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        avg_vol = avg_volume_aligned[i]
        
        # Weekly trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_val > 25
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + strong weekly trend
            if (price > donch_high and 
                volume_confirm and 
                trend_filter):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + volume + strong weekly trend
            elif (price < donch_low and 
                  volume_confirm and 
                  trend_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR volume drops significantly
            if (price < donch_low or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR volume drops significantly
            if (price > donch_high or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_ADX_Donchian_Breakout_Volume_v1"
timeframe = "1d"
leverage = 1.0