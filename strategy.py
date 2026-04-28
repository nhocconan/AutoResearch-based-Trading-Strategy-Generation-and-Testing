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
    
    # Get weekly data for trend filter and regime detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR(14) for volatility regime
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_1w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily Donchian channels (20-period) for breakout signals
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(10) for position sizing and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Precompute day-of-week filter (avoid Monday gaps, trade Tue-Fri)
    day_of_week = pd.DatetimeIndex(prices["open_time"]).dayofweek  # Monday=0, Sunday=6
    trade_day = (day_of_week >= 1) & (day_of_week <= 4)  # Tuesday to Friday
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(atr_10[i])):
            signals[i] = 0.0
            continue
        
        # Trade only Tuesday-Friday to avoid weekend gaps
        if not trade_day[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volatility regime filter: avoid extremely low volatility days
        vol_normal = atr_10[i] > (atr_1w_aligned[i] * 0.1)
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous high
        breakout_down = close[i] < low_20[i-1]  # Break below previous low
        
        # Long conditions: uptrend + normal volatility + breakout up
        long_condition = uptrend and vol_normal and breakout_up
        
        # Short conditions: downtrend + normal volatility + breakout down
        short_condition = downtrend and vol_normal and breakout_down
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout or volatility spike
        elif position == 1 and (close[i] < low_20[i-1] or atr_10[i] > (atr_1w_aligned[i] * 0.3)):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > high_20[i-1] or atr_10[i] > (atr_1w_aligned[i] * 0.3)):
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

name = "1d_WeeklyEMA20_Trend_Donchian20_Breakout_VolatilityFilter"
timeframe = "1d"
leverage = 1.0