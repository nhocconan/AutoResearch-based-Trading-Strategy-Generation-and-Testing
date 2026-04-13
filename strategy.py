#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w ADX trend filter + 1d Donchian breakout + volume confirmation
# Long: Price breaks above 1d Donchian upper (20-period) + weekly ADX > 25 + volume > 1.5x avg volume
# Short: Price breaks below 1d Donchian lower (20-period) + weekly ADX > 25 + volume > 1.5x avg volume
# Exit: Price crosses back below/above Donchian middle (10-period) or ADX drops below 20
# Uses weekly trend filter to avoid counter-trend trades, daily breakout for entry, volume for confirmation
# Target: 20-50 total trades over 4 years (5-12.5/year) for 1d timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian channels (20-period for bands, 10-period for middle/exit)
    donchian_high_20 = np.full(len(high_1d), np.nan)
    donchian_low_20 = np.full(len(low_1d), np.nan)
    donchian_high_10 = np.full(len(high_1d), np.nan)
    donchian_low_10 = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high_20[i] = np.max(high_1d[i-20:i])
        donchian_low_20[i] = np.min(low_1d[i-20:i])
    
    for i in range(10, len(high_1d)):
        donchian_high_10[i] = np.max(high_1d[i-10:i])
        donchian_low_10[i] = np.min(low_1d[i-10:i])
    
    # 1d average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on weekly data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values
        atr = np.full(len(tr), np.nan)
        plus_dm_smooth = np.full(len(plus_dm), np.nan)
        minus_dm_smooth = np.full(len(minus_dm), np.nan)
        
        # Initial averages (first period)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
            
            # Wilder's smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(len(atr), np.nan)
        minus_di = np.full(len(atr), np.nan)
        dx = np.full(len(atr), np.nan)
        
        for i in range(period, len(atr)):
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX (smoothed DX)
        adx = np.full(len(dx), np.nan)
        if len(dx) >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align indicators to 1d timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_high_10_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_10)
    donchian_low_10_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_10)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)  # Already 1d, but keep for consistency
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(donchian_high_10_aligned[i]) or np.isnan(donchian_low_10_aligned[i]) or
            np.isnan(avg_volume_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_aligned[i]
        upper_20 = donchian_high_20_aligned[i]
        lower_20 = donchian_low_20_aligned[i]
        upper_10 = donchian_high_10_aligned[i]
        lower_10 = donchian_low_10_aligned[i]
        adx = adx_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Trend filter: weekly ADX > 25 indicates strong trend
        strong_trend = adx > 25
        # Weak trend filter: ADX < 20 for exit
        weak_trend = adx < 20
        
        if position == 0:
            # Long: break above 20-period Donchian high + strong trend + volume confirmation
            if (price > upper_20 and 
                strong_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below 20-period Donchian low + strong trend + volume confirmation
            elif (price < lower_20 and 
                  strong_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 10-period Donchian low OR weak trend
            if (price < lower_10 or
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 10-period Donchian high OR weak trend
            if (price > upper_10 or
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_ADX_Donchian_Volume"
timeframe = "1d"
leverage = 1.0