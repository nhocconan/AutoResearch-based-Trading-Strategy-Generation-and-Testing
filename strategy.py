#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA(34) for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for intermediate context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low_20)
    
    # Calculate 6h volume moving average for confirmation
    vol_ma_6h = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly and daily EMA alignment
        weekly_bullish = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
        weekly_bearish = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        daily_bullish = close[i] > ema_34_1d_aligned[i]
        daily_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: current volatility above average
        vol_filter = atr_14_1d_aligned[i] > 0 and atr_14_1d_aligned[i] > atr_14_1d_aligned[i-1] * 1.1
        
        # Volume filter: current 6h volume above average
        volume_filter = vol_ma_6h_aligned[i] > 0 and volume[i] > vol_ma_6h_aligned[i] * 1.2
        
        # Breakout signals: price breaks 6h Donchian channels
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Long conditions: weekly bullish, daily bullish, volume, volatility, upward breakout
        long_condition = (weekly_bullish and 
                         daily_bullish and 
                         vol_filter and 
                         volume_filter and 
                         breakout_up)
        
        # Short conditions: weekly bearish, daily bearish, volume, volatility, downward breakout
        short_condition = (weekly_bearish and 
                          daily_bearish and 
                          vol_filter and 
                          volume_filter and 
                          breakout_down)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal on weekly or daily
        elif position == 1 and (not weekly_bullish or not daily_bullish):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not weekly_bearish or not daily_bearish):
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

name = "6h_WeeklyDailyEMA_Trend_Breakout_VolVolFilter"
timeframe = "6h"
leverage = 1.0