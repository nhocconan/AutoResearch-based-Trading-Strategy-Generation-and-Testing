#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_1dVolumeSpike_1dTrend
Weekly Donchian breakout with daily volume spike and trend filter:
- Long when price breaks above weekly Donchian high + daily volume spike + price above daily EMA200
- Short when price breaks below weekly Donchian low + daily volume spike + price below daily EMA200
- Exit when price crosses back below/above daily EMA200
- Uses weekly Donchian (20 periods) for structure, daily volume spike for conviction, daily EMA200 for trend filter
- Designed for 10-25 trades/year per symbol
Works in both bull (captures breakouts) and bear (short breakdowns) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window=20):
    """Calculate Donchian channels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(window-1, n):
        upper[i] = np.max(high[i-window+1:i+1])
        lower[i] = np.min(low[i-window+1:i+1])
    return upper, lower

def calculate_ema(arr, span):
    """Calculate EMA with proper handling."""
    if len(arr) < span:
        return np.full(len(arr), np.nan)
    s = pd.Series(arr)
    ema = s.ewm(span=span, adjust=False, min_periods=span).mean()
    return ema.values

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly Donchian channels (20 periods)
    donchian_upper_weekly, donchian_lower_weekly = calculate_donchian(high_weekly, low_weekly, window=20)
    
    # Align weekly Donchian to daily timeframe
    donchian_upper_daily = align_htf_to_ltf(prices, df_weekly, donchian_upper_weekly)
    donchian_lower_daily = align_htf_to_ltf(prices, df_weekly, donchian_lower_weekly)
    
    # Calculate daily EMA200 for trend filter
    ema200 = calculate_ema(close, 200)
    
    # Calculate daily volume spike (volume > 2x 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need sufficient data for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_daily[i]) or np.isnan(donchian_lower_daily[i]) or
            np.isnan(ema200[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian high + volume spike + price above EMA200
            if (close[i] > donchian_upper_daily[i] and 
                volume_spike[i] and 
                close[i] > ema200[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low + volume spike + price below EMA200
            elif (close[i] < donchian_lower_daily[i] and 
                  volume_spike[i] and 
                  close[i] < ema200[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA200
            if close[i] < ema200[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA200
            if close[i] > ema200[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchianBreakout_1dVolumeSpike_1dTrend"
timeframe = "1d"
leverage = 1.0