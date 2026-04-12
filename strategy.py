#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
    # Trade only in direction of 1d ADX > 25 to avoid chop whipsaws
    # Volume spike (>1.5x 20-period average) confirms participation
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    # Works in bull/bear markets by only trading when 1d trend is strong (ADX>25)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous 1d bar's Donchian channels (20-period)
    # Upper = max(high_1d[-21:-1]), Lower = min(low_1d[-21:-1])
    lookback = 20
    donchian_upper = np.full(len(df_1d), np.nan)
    donchian_lower = np.full(len(df_1d), np.nan)
    
    for i in range(lookback, len(df_1d)):
        donchian_upper[i] = np.max(high_1d[i-lookback:i])
        donchian_lower[i] = np.min(low_1d[i-lookback:i])
    
    # Calculate 1d ADX (14-period) for trend filter
    # ADX requires +DM, -DM, TR, then DX, then smoothed ADX
    period = 14
    tr = np.zeros(len(df_1d))
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM
    atr = np.zeros(len(df_1d))
    plus_di = np.zeros(len(df_1d))
    minus_di = np.zeros(len(df_1d))
    dx = np.zeros(len(df_1d))
    adx = np.zeros(len(df_1d))
    
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_smooth = np.mean(plus_dm[1:period+1])
    minus_dm_smooth = np.mean(minus_dm[1:period+1])
    
    for i in range(period+1, len(df_1d)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_smooth = (plus_dm_smooth * (period-1) + plus_dm[i]) / period
        minus_dm_smooth = (minus_dm_smooth * (period-1) + minus_dm[i]) / period
        
        plus_di[i] = 100 * plus_dm_smooth / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_smooth / atr[i] if atr[i] != 0 else 0
        
        dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        if i < 2*period:
            adx[i] = np.mean(dx[period:i+1])
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Get 1d volume for confirmation (>1.5x 20-period average)
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align all indicators to LTF (6h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # 1d trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Entry logic: Breakout + strong trend + volume confirmation
        long_entry = long_breakout and strong_trend and volume_spike_aligned[i]
        short_entry = short_breakout and strong_trend and volume_spike_aligned[i]
        
        # Exit logic: opposite breakout or trend weakness
        long_exit = short_breakout or not strong_trend
        short_exit = long_breakout or not strong_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_breakout_adx25_volume_v1"
timeframe = "6h"
leverage = 1.0