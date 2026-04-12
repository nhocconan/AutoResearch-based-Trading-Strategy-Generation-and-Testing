#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and 1w trend filter
    # Donchian(20) breakout captures strong momentum moves
    # 1d volume > 1.5x 20-period average confirms institutional participation
    # 1w EMA50 filter ensures we only trade in direction of weekly trend
    # Works in bull/bear by adapting to weekly trend context
    # Target: 12-37 trades/year per symbol (50-150 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x daily average volume
        # Approximate: 2x 12h bars per day, so compare to half of daily MA
        vol_ratio = volume[i] / (vol_ma_20_aligned[i] / 2.0)
        
        # Donchian breakout signals with volume confirmation and trend filter
        breakout_long = (close[i] > high_20_aligned[i]) and (vol_ratio > 1.5) and (close[i] > ema_50_1w_aligned[i])
        breakout_short = (close[i] < low_20_aligned[i]) and (vol_ratio > 1.5) and (close[i] < ema_50_1w_aligned[i])
        
        # Exit conditions: return to opposite Donchian band
        long_exit = close[i] < low_20_aligned[i]
        short_exit = close[i] > high_20_aligned[i]
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
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

name = "12h_1d_1w_donchian_breakout_vol_trend_v1"
timeframe = "12h"
leverage = 1.0