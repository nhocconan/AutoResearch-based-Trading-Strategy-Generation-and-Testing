#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy combining weekly Donchian breakouts with monthly EMA trend filter.
# Long when price breaks above weekly Donchian upper with monthly EMA alignment (price > EMA) and volume confirmation.
# Short when price breaks below weekly Donchian lower with monthly EMA alignment (price < EMA) and volume confirmation.
# Exit when price returns to weekly Donchian midpoint or monthly EMA slope changes direction.
# Uses weekly Donchian for structure, monthly EMA for trend filter, volume for confirmation.
# Target: 10-20 trades/year per symbol (40-80 total over 4 years) to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-week)
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Load monthly data ONCE for EMA trend filter
    df_1M = get_htf_data(prices, '1M')
    if len(df_1M) < 20:
        return np.zeros(n)
    
    close_1M = df_1M['close'].values
    
    # Calculate monthly EMA(20)
    ema_1M = pd.Series(close_1M).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to lower timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    ema_1M_aligned = align_htf_to_ltf(prices, df_1M, ema_1M)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(ema_1M_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below monthly EMA
        price_above_ema = close[i] > ema_1M_aligned[i]
        price_below_ema = close[i] < ema_1M_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian upper AND price above monthly EMA
            if (close[i] > donchian_upper_aligned[i] and 
                price_above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower AND price below monthly EMA
            elif (close[i] < donchian_lower_aligned[i] and 
                  price_below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or price crosses below monthly EMA
            if (close[i] <= donchian_mid_aligned[i] or 
                close[i] < ema_1M_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or price crosses above monthly EMA
            if (close[i] >= donchian_mid_aligned[i] or 
                close[i] > ema_1M_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyDonchian_MonthlyEMA_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0