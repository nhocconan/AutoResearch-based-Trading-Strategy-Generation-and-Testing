#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation.
# Uses daily price channel breakouts for trend continuation, weekly EMA for long-term trend filter,
# and volume spike for confirmation. Works in bull markets (breakouts above upper band) and bear
# markets (breakdowns below lower band). Target: 10-25 trades/year to minimize fee drag.
name = "1d_Donchian20_WeeklyEMA200_VolumeBreakout"
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
    
    # Get weekly data for EMA200 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(200) for trend filter
    ema_200_weekly = pd.Series(df_weekly['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_weekly, ema_200_weekly)
    
    # Calculate daily Donchian channels (20-period)
    # Use rolling window on daily high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period EMA (high threshold to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 days for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_200_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + weekly uptrend + volume spike
            if (price > donchian_upper[i] and price > ema_200_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + weekly downtrend + volume spike
            elif (price < donchian_lower[i] and price < ema_200_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below lower Donchian or trend reverses
            if price < donchian_lower[i] or price < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above upper Donchian or trend reverses
            if price > donchian_upper[i] or price > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals