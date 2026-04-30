#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation (>1.5x 20-bar avg).
# Uses Donchian channels for structure, 1d EMA200 for higher timeframe trend filter.
# Volume confirmation reduces false breakouts. Session filter (08-20 UTC) avoids low-liquidity periods.
# Discrete position sizing at ±0.25 to balance return and fee drag.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag on 4h timeframe.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion exits when price retests channel midpoint.

name = "4h_Donchian20_1dEMA200_Trend_VolumeConfirm_Session_v1"
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
    
    # Load 1d data ONCE before loop for EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d_vals = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d_vals).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_200_1d = ema_200_1d_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid = donchian_mid[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, close > 1d EMA200, volume spike, in session
            if (curr_close > curr_upper and 
                curr_close > curr_ema_200_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, close < 1d EMA200, volume spike, in session
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_200_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves back below Donchian midpoint
            if curr_close < curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price moves back above Donchian midpoint
            if curr_close > curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals