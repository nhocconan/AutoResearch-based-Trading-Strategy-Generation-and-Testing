#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.5x 20-bar 12h volume average.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume confirmation.
# Uses discrete sizing 0.25 to minimize fee churn. Designed to capture trends in both bull and bear markets.

name = "12h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Load 12h data ONCE before loop for Donchian and volume calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    vol_12h = df_12h['volume'].values
    
    # Donchian(20) channels on 12h data
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    highest_high_20_aligned = align_htf_to_ltf(prices, df_12h, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_20)
    
    # Volume MA(20) on 12h data
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
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
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_donchian_high = highest_high_20_aligned[i]
        curr_donchian_low = lowest_low_20_aligned[i]
        curr_vol_ma = vol_ma_20_aligned[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price > Donchian(20) high AND price > 1d EMA50 AND volume confirmation
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian(20) low AND price < 1d EMA50 AND volume confirmation
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian(20) low OR price < 1d EMA50 (trend violation)
            if (curr_close < curr_donchian_low or 
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian(20) high OR price > 1d EMA50 (trend violation)
            if (curr_close > curr_donchian_high or 
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals