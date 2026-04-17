#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h EMA trend filter + 1h Donchian(20) breakout + volume confirmation + session filter (08-20 UTC).
Long when price breaks above 20-period high with 4h EMA50 > EMA200 (uptrend) and volume > 1.5x 20-period volume average.
Short when price breaks below 20-period low with 4h EMA50 < EMA200 (downtrend) and volume > 1.5x 20-period volume average.
Uses discrete position sizing 0.20 to limit fee drag. Target: 60-150 total trades over 4 years.
Session filter reduces noise trades outside active market hours. Donchian breakouts capture structural moves;
4h EMA filter ensures trading with higher timeframe trend; volume confirms participation. Designed to work in
bull markets (breakout continuation) and bear markets (strong trend continuation).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 and EMA200 for trend
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_50_4h = ema(close_4h, 50)
    ema_200_4h = ema(close_4h, 200)
    
    # Calculate 1h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 1h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (1h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Session filter: 08-20 UTC (active market hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_4h_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
        uptrend = ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        downtrend = ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with uptrend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                uptrend and 
                volume_confirmed):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-period low with downtrend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-period low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 20-period high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA50_200_Donchian20_Breakout_Volume_Confirm_Session"
timeframe = "1h"
leverage = 1.0