#!/usr/bin/env python3
"""
12h 1w/1d Price Channel Breakout + Volume Spike + ADX Trend Filter
Hypothesis: Combines weekly and daily trend filters with 12h price channel breakouts (using Donchian channels) and volume confirmation. 
ADX ensures we only trade in trending markets, reducing false signals in ranging conditions. 
Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag while capturing major moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(high, np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX with proper smoothing"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # Initial value
        result[period-1] = np.nansum(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, period)
    plus_di = 100 * wilders_smooth(plus_dm, period) / np.where(atr != 0, atr, 1)
    minus_di = 100 * wilders_smooth(minus_dm, period) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for intermediate trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA on weekly for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w)
    ema_1w[0] = close_1w[0]
    alpha = 2 / (50 + 1)  # 50-period EMA
    for i in range(1, len(close_1w)):
        ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate EMA on daily for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d)
    ema_1d[0] = close_1d[0]
    alpha = 2 / (20 + 1)  # 20-period EMA
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels on 12h data
    upper_channel, lower_channel = calculate_donchian_channels(high, low, period=20)
    
    # Calculate ADX on 12h data
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_1w_val = ema_1w_aligned[i]
        ema_1d_val = ema_1d_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        adx_val = adx[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel + both EMAs above price (uptrend) + ADX > 25 + volume spike
            if (close[i] > upper_val and 
                ema_1w_val > close[i] and 
                ema_1d_val > close[i] and 
                adx_val > 25 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel + both EMAs below price (downtrend) + ADX > 25 + volume spike
            elif (close[i] < lower_val and 
                  ema_1w_val < close[i] and 
                  ema_1d_val < close[i] and 
                  adx_val > 25 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower channel or ADX weakens
            if close[i] < lower_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper channel or ADX weakens
            if close[i] > upper_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PriceChannelBreakout_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0