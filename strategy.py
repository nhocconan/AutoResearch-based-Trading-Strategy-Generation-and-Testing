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
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian channels (25-period) - slightly longer for fewer trades
    donchian_high_25 = pd.Series(close_1d).rolling(window=25, min_periods=25).max().values
    donchian_low_25 = pd.Series(close_1d).rolling(window=25, min_periods=25).min().values
    donchian_high_25_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_25)
    donchian_low_25_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_25)
    
    # Calculate daily ATR(20) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily volume average for volume confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_25_aligned[i]) or 
            np.isnan(donchian_low_25_aligned[i]) or 
            np.isnan(atr_20_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA50 for long, below for short
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: ATR not too low (avoid choppy markets)
        vol_ok = atr_20_1d_aligned[i] > (np.nanmean(atr_20_1d_aligned[max(0, i-20):i+1]) * 0.5)
        
        # Volume filter: current volume above 20-period average
        vol_ok_volume = volume[i] > vol_ma_20_1d_aligned[i]
        
        # Breakout conditions - using 25-period Donchian for fewer, stronger signals
        long_breakout = close[i] > donchian_high_25_aligned[i]
        short_breakout = close[i] < donchian_low_25_aligned[i]
        
        # Long conditions: uptrend + volatility ok + volume + long breakout
        long_condition = uptrend and vol_ok and vol_ok_volume and long_breakout
        
        # Short conditions: downtrend + volatility ok + volume + short breakout
        short_condition = downtrend and vol_ok and vol_ok_volume and short_breakout
        
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

name = "4h_Donchian25_EMA50_VolumeFilter_Session"
timeframe = "4h"
leverage = 1.0