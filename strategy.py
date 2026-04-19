#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Uses weekly Donchian channels to determine long-term trend direction.
# Enters long when price breaks above 20-day high in uptrend, short when breaks below 20-day low in downtrend.
# Volume confirmation reduces false breakouts. Designed for low trade frequency (target: 15-25/year).
name = "1d_Donchian20_1wTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Daily Donchian breakout levels (20-period)
    donchian_high_daily = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_daily = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Weekly trend direction
        weekly_uptrend = donchian_high_aligned[i] > donchian_low_aligned[i]  # Always true, but we need direction
        # Better: use price position relative to weekly midpoint
        weekly_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        weekly_uptrend = price > weekly_mid
        
        if position == 0:
            # Long: price breaks above daily Donchian high, weekly uptrend, volume confirmation
            if (price > donchian_high_daily[i] and weekly_uptrend and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low, weekly downtrend, volume confirmation
            elif (price < donchian_low_daily[i] and not weekly_uptrend and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below daily Donchian low or weekly trend turns down
            if price < donchian_low_daily[i] or price < weekly_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above daily Donchian high or weekly trend turns up
            if price > donchian_high_daily[i] or price > weekly_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals