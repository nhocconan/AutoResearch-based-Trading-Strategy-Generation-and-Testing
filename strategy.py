#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses tight volume threshold (2.0x average) to limit trades to ~100 total over 4 years.
# Only enters when price breaks Donchian upper/lower band with volume confirmation and 1d EMA50 trend alignment.
# Designed for low trade frequency to avoid fee drag. Works in bull/bear via 1d EMA50 trend filter.

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    # We need to calculate these on the 4h data itself
    df_4h = prices  # since we're on 4h timeframe
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 2.0x 20-period average (tight threshold to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, 1d EMA50 uptrend, volume spike confirmation
            if (curr_close > curr_high_20 and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower band, 1d EMA50 downtrend, volume spike confirmation
            elif (curr_close < curr_low_20 and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian lower band or reverses below entry
            if curr_close < curr_low_20 or curr_close < entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian upper band or reverses above entry
            if curr_close > curr_high_20 or curr_close > entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals