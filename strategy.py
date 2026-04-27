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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for volume and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily average volume (20-period)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    
    # Calculate daily ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(avg_volume_20_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above weekly EMA20 for long, below for short
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume filter: current volume > 1.5x average volume
        volume_ok = volume[i] > (avg_volume_20_aligned[i] * 1.5)
        
        # Volatility filter: ATR not too low (avoid choppy markets)
        atr_ma_20 = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
        atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
        if np.isnan(atr_ma_20_aligned[i]):
            signals[i] = 0.0
            continue
        vol_ok = atr_14_1d_aligned[i] > (atr_ma_20_aligned[i] * 0.5)
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_20_aligned[i]
        short_breakout = close[i] < donchian_low_20_aligned[i]
        
        # Long conditions: weekly uptrend + volume ok + volatility ok + long breakout
        long_condition = weekly_uptrend and volume_ok and vol_ok and long_breakout
        
        # Short conditions: weekly downtrend + volume ok + volatility ok + short breakout
        short_condition = weekly_downtrend and volume_ok and vol_ok and short_breakout
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite breakout
        elif position == 1 and short_breakout:
            signals[i] = 0.0
            position = 0
        elif position == -1 and long_breakout:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyEMA20_VolumeFilter_DonchianBreakout"
timeframe = "12h"
leverage = 1.0