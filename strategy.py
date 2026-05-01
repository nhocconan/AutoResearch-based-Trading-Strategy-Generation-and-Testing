#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long: price breaks above Donchian upper channel AND price > 1w EMA50 AND volume > 2x 20-bar average.
# Short: price breaks below Donchian lower channel AND price < 1w EMA50 AND volume > 2x 20-bar average.
# Donchian channels capture volatility-based breakouts; 1w EMA50 ensures alignment with higher timeframe trend.
# Volume confirmation filters weak breakouts. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 12-37 trades/year on 12h (50-150 total over 4 years). Discrete sizing 0.25 to minimize fee drag.

name = "12h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    
    # Align Donchian channels to 12h primary timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h primary timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian (20) + EMA50 (50)
    
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
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_donchian_upper = donchian_upper_aligned[i]
        curr_donchian_lower = donchian_lower_aligned[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Volume confirmation: current 12h volume > 2x 20-period average
        vol_12h = df_12h['volume'].values
        vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
        curr_vol_ma = vol_ma_12h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper AND price > 1w EMA50 AND volume confirmation
            if (curr_close > curr_donchian_upper and 
                curr_close > curr_ema_50_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < 1w EMA50 AND volume confirmation
            elif (curr_close < curr_donchian_lower and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR price < 1w EMA50 (trend violation)
            if (curr_close < curr_donchian_lower or 
                curr_close < curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR price > 1w EMA50 (trend violation)
            if (curr_close > curr_donchian_upper or 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals