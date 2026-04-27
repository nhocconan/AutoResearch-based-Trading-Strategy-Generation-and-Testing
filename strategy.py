#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly ADX trend filter and volume confirmation.
# Uses 20-day Donchian channels for breakout signals, weekly ADX(14) > 25 to filter for trending markets,
# and volume spikes (1.5x 20-day average) to confirm breakouts. Works in both bull and bear
# markets by only taking breakouts in the direction of the weekly trend. Target: 10-20 trades/year
# to minimize fee decay while capturing major trend moves. Focus on BTC/ETH.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    n_1d = len(high_1d)
    
    upper_channel = np.full(n_1d, np.nan)
    lower_channel = np.full(n_1d, np.nan)
    
    for i in range(19, n_1d):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate ADX(14) on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    n_1w = len(high_1w)
    
    # True Range
    tr = np.zeros(n_1w)
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, n_1w):
        tr[i] = max(high_1w[i] - low_1w[i], 
                   abs(high_1w[i] - close_1w[i-1]),
                   abs(low_1w[i] - close_1w[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n_1w)
    minus_dm = np.zeros(n_1w)
    for i in range(1, n_1w):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.sum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    # DI values
    plus_di = np.full(n_1w, np.nan)
    minus_di = np.full(n_1w, np.nan)
    dx = np.full(n_1w, np.nan)
    
    for i in range(14, n_1w):
        if tr14[i] > 0:
            plus_di[i] = 100 * (plus_dm14[i] / tr14[i])
            minus_di[i] = 100 * (minus_dm14[i] / tr14[i])
            dx[i] = 100 * (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]))
    
    # ADX
    adx = np.full(n_1w, np.nan)
    if len(dx) >= 28:  # Need 14 for initial ADX + 14 more for smoothing
        adx[27] = np.mean(dx[14:28])
        for i in range(28, n_1w):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 20-day average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align indicators to lower timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(30, 30) + 20  # Donchian needs 20, ADX needs ~30, vol needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx_aligned[i] > 25
        
        if position == 0 and is_trending and volume_confirmation:
            # Long: Price breaks above upper Donchian channel
            if price > upper_channel_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Donchian channel
            elif price < lower_channel_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below lower Donchian channel
            if price < lower_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above upper Donchian channel
            if price > upper_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
        else:
            signals[i] = 0.0
    
    return signals

name = "DailyDonchian20_WeeklyADX25_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0