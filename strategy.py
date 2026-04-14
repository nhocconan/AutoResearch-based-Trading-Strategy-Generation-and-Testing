#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian Breakout with Weekly Volume Confirmation and Weekly Trend Filter
# Takes long when price breaks above daily Donchian upper band with weekly volume spike and weekly EMA > weekly EMA(50)
# Takes short when price breaks below daily Donchian lower band with weekly volume spike and weekly EMA < weekly EMA(50)
# Exits when price crosses back below/above the daily Donchian midline
# Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate daily Donchian channels (20-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    donchian_high = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly volume average (20-period)
    vol_weekly = df_weekly['volume'].values
    vol_ma_weekly = pd.Series(vol_weekly).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_daily, donchian_mid)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    vol_ma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_weekly)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Donchian and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_daily_current = volume[i]  # Current daily volume
        
        if position == 0:
            # Long setup: break above Donchian high with volume spike and bullish trend
            if (price > donchian_high_aligned[i] and 
                vol_daily_current > 1.5 * vol_ma_weekly_aligned[i] and  # Volume spike
                ema_weekly_aligned[i] > 0):                          # Weekly trend filter
                # Additional check: price above weekly EMA for trend confirmation
                if price > ema_weekly_aligned[i]:
                    position = 1
                    signals[i] = position_size
            # Short setup: break below Donchian low with volume spike and bearish trend
            elif (price < donchian_low_aligned[i] and 
                  vol_daily_current > 1.5 * vol_ma_weekly_aligned[i] and  # Volume spike
                  ema_weekly_aligned[i] > 0):                           # Weekly trend filter check
                # Additional check: price below weekly EMA for trend confirmation
                if price < ema_weekly_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian mid
            if price < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian mid
            if price > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_Breakout_WeeklyVolume_EMA"
timeframe = "1d"
leverage = 1.0