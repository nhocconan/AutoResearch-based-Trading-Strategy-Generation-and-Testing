#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w ADX trend filter and volume confirmation
# Works in bull markets by capturing breakouts, in bear markets by avoiding false signals via ADX
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Uses daily volume confirmation to ensure breakout validity

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX on weekly
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros(len(high))
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_sum = np.nansum(plus_dm[1:period+1])
        minus_dm_sum = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = (plus_dm_sum * (period-1) + plus_dm[i]) / period
            minus_dm_sum = (minus_dm_sum * (period-1) + minus_dm[i]) / period
            plus_di[i] = 100 * plus_dm_sum / atr[i]
            minus_di[i] = 100 * minus_dm_sum / atr[i]
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(high))
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 20-period average volume on daily
    volume_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        volume_ma_1d[i] = np.mean(volume_1d[i-20:i])
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channels on 12h
    high_20 = np.full(len(close_12h), np.nan)
    low_20 = np.full(len(close_12h), np.nan)
    for i in range(20, len(close_12h)):
        high_20[i] = np.max(high_12h[i-20:i])
        low_20[i] = np.min(low_12h[i-20:i])
    
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x daily average volume
        # Approximate current 12h volume by summing last 48 hours of 15m data (4 periods)
        vol_sum_12h = volume[i-3:i+1].sum() if i >= 3 else 0
        vol_confirmed = vol_sum_12h > 1.5 * volume_ma_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        # Entry conditions: breakout with volume confirmation and strong trend
        long_entry = long_breakout and vol_confirmed and strong_trend
        short_entry = short_breakout and vol_confirmed and strong_trend
        
        # Exit conditions: opposite breakout or trend weakening
        exit_long = position == 1 and (short_breakout or adx_1w_aligned[i] < 20)
        exit_short = position == -1 and (long_breakout or adx_1w_aligned[i] < 20)
        
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

name = "12h_1w_adx_volume_breakout"
timeframe = "12h"
leverage = 1.0