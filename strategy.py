#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Uses 1d Donchian channels for structure, 1w EMA200 for higher timeframe trend filter.
# Volume confirmation (>1.5x 50-bar avg) reduces false breakouts.
# Session filter (08-20 UTC) avoids low-liquidity periods.
# Discrete position sizing at ±0.25 to balance return and fee drag.
# Target: 50-100 total trades over 4 years (12-25/year) to avoid fee drag on 1d timeframe.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion exits when price retests Donchian levels.

name = "1d_Donchian20_1wEMA200_Trend_VolumeConfirm_Session_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    
    # Volume confirmation: volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_200_1w = ema_200_1w_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, close > 1w EMA200, volume spike, in session
            if (curr_close > curr_upper and 
                curr_close > curr_ema_200_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, close < 1w EMA200, volume spike, in session
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_200_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves back below upper Donchian (mean reversion)
            if curr_close < curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price moves back above lower Donchian (mean reversion)
            if curr_close > curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals