#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ATR(14) moving average for regime detection
    atr_ma_1d = pd.Series(atr14_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align daily ATR to 1d timeframe (no shift needed as we're already on 1d)
    atr14_aligned = atr14_1d
    atr_ma_aligned = atr_ma_1d
    
    # Calculate daily EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Donchian channels (20-period)
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume average (20-period)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr14_aligned[i]) or 
            np.isnan(atr_ma_aligned[i]) or
            np.isnan(ema50_1d[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: daily ATR above its 50-period average (avoid low volatility periods)
        vol_regime = atr14_aligned[i] > atr_ma_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume_1d[i] > vol_avg[i]
        
        # Trend filters: price above/below daily and weekly EMA50
        uptrend = close_1d[i] > ema50_1d[i] and close_1d[i] > ema50_1w_aligned[i]
        downtrend = close_1d[i] < ema50_1d[i] and close_1d[i] < ema50_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close_1d[i] > highest_high[i]
        short_breakout = close_1d[i] < lowest_low[i]
        
        # Entry conditions: Donchian breakout with volatility regime, volume, and trend alignment
        long_entry = long_breakout and vol_regime and vol_filter and uptrend
        short_entry = short_breakout and vol_regime and vol_filter and downtrend
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        long_exit = close_1d[i] < lowest_low[i] or not uptrend
        short_exit = close_1d[i] > highest_high[i] or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DailyATR_VolumeRegime_Donchian20_DailyWeeklyEMA50_Trend_v1"
timeframe = "1d"
leverage = 1.0