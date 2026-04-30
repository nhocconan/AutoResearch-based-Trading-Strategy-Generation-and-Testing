#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# Uses tight volume threshold (2.0x average) and requires breakout of 20-period Donchian channel
# with 12h EMA50 alignment. Designed for low trade frequency (~60-100 total trades over 4 years)
# to avoid fee drag. Works in bull/bear via 12h EMA50 trend filter which adapts to market regime.

name = "6h_Donchian20_12hEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Highest high and lowest low over past 20 periods (including current)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-19:i+1])  # 20 periods: i-19 to i
        lowest_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Volume confirmation: volume > 2.0x 20-period average (moderate threshold)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, 12h EMA50 uptrend, volume spike
            if (curr_close > curr_highest_high and 
                curr_close > curr_ema_50_12h and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower band, 12h EMA50 downtrend, volume spike
            elif (curr_close < curr_lowest_low and 
                  curr_close < curr_ema_50_12h and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower band or reverses below entry
            if curr_close < curr_lowest_low or curr_close < entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band or reverses above entry
            if curr_close > curr_highest_high or curr_close > entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals