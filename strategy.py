#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian Channel Breakout with Weekly Trend Filter
# Uses 20-day Donchian breakouts for trend following - proven to work in both bull and bear markets
# Weekly EMA (50) provides trend filter to avoid counter-trend trades
# Volume confirmation ensures breakout validity
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# Daily timeframe reduces trade frequency while capturing major trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (50) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume ratio (current vs 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of weekly EMA
        above_weekly_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume confirmation and uptrend filter
            if price > high_20[i] and vol_ratio[i] > 1.5 and above_weekly_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian band with volume confirmation and downtrend filter
            elif price < low_20[i] and vol_ratio[i] > 1.5 and not above_weekly_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches lower Donchian band or trend changes
            if price < low_20[i] or price < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches upper Donchian band or trend changes
            if price > high_20[i] or price > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "DailyDonchian_WeeklyEMA_Volume"
timeframe = "1d"
leverage = 1.0