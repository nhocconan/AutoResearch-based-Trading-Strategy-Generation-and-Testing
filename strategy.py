#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > 1d EMA50) and volume spike confirmation.
# Breakouts capture momentum in both bull and bear markets. 1d EMA50 ensures we trade with higher timeframe trend.
# Volume spike (current 6h volume > 1.8x 20-bar 6h average) confirms institutional participation.
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 trades over 4 years (12-37/year).

name = "6h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 6h data ONCE before loop for Donchian channels and volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # 6h Donchian channels (20-period)
    highest_high_6h = pd.Series(df_6h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_6h = pd.Series(df_6h['low'].values).rolling(window=20, min_periods=20).min().values
    highest_high_6h_aligned = align_htf_to_ltf(prices, df_6h, highest_high_6h)
    lowest_low_6h_aligned = align_htf_to_ltf(prices, df_6h, lowest_low_6h)
    
    # 6h volume MA (20-period)
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high_6h_aligned[i]) or 
            np.isnan(lowest_low_6h_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_highest_high = highest_high_6h_aligned[i]
        curr_lowest_low = lowest_low_6h_aligned[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.8x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # Breakout conditions
        bullish_breakout = curr_high > curr_highest_high
        bearish_breakout = curr_low < curr_lowest_low
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout AND price > 1d EMA50 AND volume confirmation
            if (bullish_breakout and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND price < 1d EMA50 AND volume confirmation
            elif (bearish_breakout and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < 1d EMA50 (trend violation) OR bearish breakout (contrarian signal)
            if (curr_close < curr_ema_50_1d or 
                bearish_breakout):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > 1d EMA50 (trend violation) OR bullish breakout (contrarian signal)
            if (curr_close > curr_ema_50_1d or 
                bullish_breakout):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals