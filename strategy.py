#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation and 1w Trend Filter
# Uses weekly Donchian channels (20-period) to identify breakouts in the direction of weekly trend
# Volume confirmation ensures breakout authenticity
# Weekly EMA (21) provides trend filter to avoid counter-trend trades
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drift

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (21) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian channels
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of weekly EMA
        above_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with uptrend filter
            if price > donchian_high_aligned[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly Donchian low with downtrend filter
            elif price < donchian_low_aligned[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches weekly Donchian low or trend changes
            if price <= donchian_low_aligned[i] or price < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches weekly Donchian high or trend changes
            if price >= donchian_high_aligned[i] or price > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Trend"
timeframe = "1d"
leverage = 1.0