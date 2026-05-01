#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 2x 20-day avg volume.
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 2x 20-day avg volume.
# Uses discrete sizing 0.25 to minimize fee churn. Designed to capture strong trends in both bull and bear markets.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Session filter: 00-23 UTC (full day for 1d timeframe)
        hour = hours[i]
        in_session = True  # 1d timeframe uses full day
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Donchian(20) calculation on 1d data (requires 20 periods of high/low)
        if i < 20 + start_idx:  # need extra warmup for Donchian
            signals[i] = 0.0
            continue
            
        # Calculate Donchian channels: upper = highest high, lower = lowest low
        highest_high = np.max(high[i-19:i+1])  # 20 periods including current
        lowest_low = np.min(low[i-19:i+1])
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        curr_vol_ma = vol_ma[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price > upper Donchian AND price > 1w EMA50 AND volume confirmation
            if (curr_close > highest_high and 
                curr_close > curr_ema_50_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price < lower Donchian AND price < 1w EMA50 AND volume confirmation
            elif (curr_close < lowest_low and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < lower Donchian (trend violation) OR price < 1w EMA50
            if (curr_close < lowest_low or 
                curr_close < curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > upper Donchian (trend violation) OR price > 1w EMA50
            if (curr_close > highest_high or 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals