#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA200 trend filter and volume spike.
# Uses weekly EMA for trend direction and daily Donchian breakout for entry timing.
# Designed to capture long-term trends while avoiding false breakouts in ranging markets.
# Target: 10-25 trades per year to minimize fee drag and work in both bull and bear markets.
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(200) for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Use rolling window on daily high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2x 20-period EMA (high threshold for significance)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 days for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + weekly uptrend + volume spike
            if (price > donchian_high[i] and price > ema_200_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + weekly downtrend + volume spike
            elif (price < donchian_low[i] and price < ema_200_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or trend reverses
            if price < donchian_low[i] or price < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or trend reverses
            if price > donchian_high[i] or price > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals