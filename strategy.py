#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Uses 12h EMA50 for trend direction and requires volume > 1.8x average for confirmation.
# Designed for low trade frequency (~80-120 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by only taking breakouts in the direction of the 12h EMA50 trend.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        # Calculate Donchian(20) levels using previous 12h bar (completed)
        if len(df_12h) >= 20:
            # Calculate Donchian levels for each 12h bar
            donchian_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
            donchian_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
            
            # Align to 4h timeframe with proper delay (wait for 12h bar to close)
            donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
            donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
        else:
            donchian_high_aligned = np.full(n, np.nan)
            donchian_low_aligned = np.full(n, np.nan)
        
        # Volume confirmation: volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.8 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, 12h EMA50 uptrend, volume spike
            if (curr_close > donchian_high_aligned[i] and 
                curr_close > curr_ema_50_12h and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian low, 12h EMA50 downtrend, volume spike
            elif (curr_close < donchian_low_aligned[i] and 
                  curr_close < curr_ema_50_12h and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low or reverses below entry
            if curr_close < donchian_low_aligned[i] or curr_close < entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high or reverses above entry
            if curr_close > donchian_high_aligned[i] or curr_close > entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals