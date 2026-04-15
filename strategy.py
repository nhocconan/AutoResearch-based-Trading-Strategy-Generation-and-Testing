#!/usr/bin/env python3
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
    
    # Get daily HTF data once before loop (6h primary, 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(daily_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = daily_high - ema_13
    bear_power = daily_low - ema_13
    
    # Smooth Elder Ray with EMA8 for signal line
    bull_power_smooth = pd.Series(bull_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Calculate 1d Weekly EMA40 for trend filter (using daily data)
    # Weekly EMA40 approximation: 40-day EMA on daily data
    weekly_ema_40 = pd.Series(daily_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    weekly_ema_40_6h = align_htf_to_ltf(prices, df_1d, weekly_ema_40)
    bull_power_smooth_6h = align_htf_to_ltf(prices, df_1d, bull_power_smooth)
    bear_power_smooth_6h = align_htf_to_ltf(prices, df_1d, bear_power_smooth)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_40_6h[i]) or np.isnan(bull_power_smooth_6h[i]) or 
            np.isnan(bear_power_smooth_6h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d trend filter: price relative to weekly EMA40
        # 2. 1d Elder Ray filter: bull/bear power alignment with trend
        # 3. 6h Donchian breakout: price breaks 20-period channel
        # 4. 6h volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: bullish alignment + Donchian breakout
        if (close[i] > weekly_ema_40_6h[i] and          # Price above weekly EMA40 (uptrend)
            bull_power_smooth_6h[i] > 0 and            # Positive bull power
            bear_power_smooth_6h[i] < 0 and            # Negative bear power (confirmation)
            close[i] > highest_20[i] and               # Donchian breakout above
            volume_ratio[i] > 1.3):                    # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: bearish alignment + Donchian breakdown
        elif (close[i] < weekly_ema_40_6h[i] and       # Price below weekly EMA40 (downtrend)
              bear_power_smooth_6h[i] < 0 and          # Negative bear power
              bull_power_smooth_6h[i] > 0 and          # Positive bull power (confirmation)
              close[i] < lowest_20[i] and              # Donchian breakdown below
              volume_ratio[i] > 1.3):                  # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_WeeklyEMA40_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0