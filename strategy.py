#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.5x 20-bar avg).
# Uses price channel breakouts for trend capture, 12h EMA50 for higher timeframe trend filter,
# volume confirmation to reduce false breakouts, and session filter (08-20 UTC) to avoid low liquidity.
# Discrete position sizing at ±0.25 to balance return and fee drag.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag on 4h timeframe.
# Works in bull markets via breakout continuation and in bear markets via quick exits when price retests channel middle.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_Session_v1"
timeframe = "4h"
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
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(middle_20[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_middle_20 = middle_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, close > 12h EMA50, volume spike, in session
            if (curr_close > curr_highest_20 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, close < 12h EMA50, volume spike, in session
            elif (curr_close < curr_lowest_20 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves back below middle of Donchian channel
            if curr_close < curr_middle_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price moves back above middle of Donchian channel
            if curr_close > curr_middle_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals