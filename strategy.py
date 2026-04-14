#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above daily Donchian upper band, weekly trend is up (price above weekly 50 EMA), and volume > 1.5x 20-day average
# Short when price breaks below daily Donchian lower band, weekly trend is down (price below weekly 50 EMA), and volume > 1.5x 20-day average
# Exit when price crosses the daily Donchian midline
# Weekly trend filter prevents counter-trend trades, targeting 30-100 trades over 4 years to minimize fee drag

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
    
    # Calculate daily Donchian channel (20-period lookback)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    donchian_upper = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate weekly 50 EMA for trend filter
    close_weekly = df_weekly['close'].values
    weekly_ema50 = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_daily, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_daily, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_daily, donchian_middle)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 20-period Donchian and 50 EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_daily_current = volume[i]  # Current daily volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume confirmation and weekly uptrend
            if (price > donchian_upper_aligned[i] and 
                vol_daily_current > 1.5 * vol_ma_daily_aligned[i] and  # Volume confirmation
                price > weekly_ema50_aligned[i]):                      # Weekly uptrend filter
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume confirmation and weekly downtrend
            elif (price < donchian_lower_aligned[i] and 
                  vol_daily_current > 1.5 * vol_ma_daily_aligned[i] and  # Volume confirmation
                  price < weekly_ema50_aligned[i]):                      # Weekly downtrend filter
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0