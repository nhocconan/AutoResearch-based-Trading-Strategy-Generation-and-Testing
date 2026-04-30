#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation (>1.5x 20-bar avg).
# Uses weekly EMA200 for higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation reduces false breakouts. Session filter (08-20 UTC) avoids low-liquidity periods.
# Discrete position sizing at ±0.30 to balance return and fee drag.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag on 1d timeframe.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion exits when price retests the Donchian mid-line.

name = "1d_Donchian20_1wEMA200_Trend_VolumeConfirm_Session_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA200
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 1d timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and EMA200
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_donchian_mid = donchian_mid[i]
        curr_ema_200_1w = ema_200_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 20-day high, close > 1w EMA200, volume spike, in session
            if (curr_close > curr_high_20 and 
                curr_close > curr_ema_200_1w and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 20-day low, close < 1w EMA200, volume spike, in session
            elif (curr_close < curr_low_20 and 
                  curr_close < curr_ema_200_1w and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves back below the Donchian mid-line
            if curr_close < curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit condition: price moves back above the Donchian mid-line
            if curr_close > curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals