#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and 1d ADX trend filter.
# Uses 4h for entry timing (breakouts above/below 20-period Donchian channels),
# 1d ADX > 25 to filter for trending markets only, and 1d volume spike (>1.5x 20-period average) 
# to confirm institutional interest. Designed to work in both bull and bear markets by 
# only taking trades in the direction of the higher timeframe trend.
# Target: 15-30 trades per year (60-120 total over 4 years) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1-day ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values using Wilder's smoothing (similar to EMA but different factor)
        def wilders_smooth(data, period):
            n = len(data)
            result = np.full(n, np.nan)
            if n < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period+1])  # Skip first NaN in data
            for i in range(period, n):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period - 1) + data[i]) / period
                else:
                    result[i] = np.nan
            return result
        
        # Smooth TR, +DM, -DM
        atr = wilders_smooth(tr, period)
        plus_dm_smooth = wilders_smooth(plus_dm, period)
        minus_dm_smooth = wilders_smooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
        minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1-day volume average (20-period)
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-20:i])
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 4-hour Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        n = len(high)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        for i in range(period-1, n):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Volume spike condition: current 1h volume > 1.5x 1d average volume
    volume_spike = volume > (1.5 * vol_avg_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    # Start from index 20 to ensure Donchian channels are calculated
    for i in range(20, n):
        # Skip if required data is not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike + trending market
            if (close[i] > donch_upper[i] and 
                volume_spike[i] and 
                trending):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower + volume spike + trending market
            elif (close[i] < donch_lower[i] and 
                  volume_spike[i] and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian lower or ADX falls below 20
            if (close[i] < donch_lower[i] or 
                adx_1d_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian upper or ADX falls below 20
            if (close[i] > donch_upper[i] or 
                adx_1d_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Donchian_Volume_ADX"
timeframe = "1h"
leverage = 1.0