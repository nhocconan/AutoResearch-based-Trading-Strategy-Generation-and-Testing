#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h EMA trend filter + 1d Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-day high with 4h EMA50 > EMA200 (uptrend) and volume > 1.5x 20-day volume average.
Short when price breaks below 20-day low with 4h EMA50 < EMA200 (downtrend) and volume > 1.5x 20-day volume average.
Uses 4h for trend direction (reduces whipsaw), 1d for structure (Donchian), 1h for precise entry timing.
Session filter (08-20 UTC) to avoid low-liquidity periods. Target signal magnitude 0.20 for risk control.
Designed to work in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h EMA50 and EMA200
    close_4h_s = pd.Series(close_4h)
    ema_50_4h = close_4h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = close_4h_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_1d, low_1d, 20)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (1h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-day volume average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with 4h uptrend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                ema_50_4h_aligned[i] > ema_200_4h_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-day low with 4h downtrend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_50_4h_aligned[i] < ema_200_4h_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 20-day high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA50_200_1dDonchian20_Breakout_Volume_Confirm_Session"
timeframe = "1h"
leverage = 1.0