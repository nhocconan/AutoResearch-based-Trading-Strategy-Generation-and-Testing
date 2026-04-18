#!/usr/bin/env python3
"""
12h_WeeklyDonchianBreakout_1dVolumeSpike_1dTrend
Weekly Donchian breakout with daily volume confirmation and trend filter:
- Long when price breaks above weekly Donchian high + daily volume spike + daily close above weekly EMA
- Short when price breaks below weekly Donchian low + daily volume spike + daily close below weekly EMA
- Exit when price breaks opposite Donchian boundary
- Uses weekly Donchian (20 weeks) as primary structure, daily for filters
- Designed for 15-30 trades/year per symbol
Works in bull markets (breakouts) and bear markets (breakdowns)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA with proper handling of NaN."""
    close_series = pd.Series(close)
    ema = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Donchian (20 weeks)
    donchian_upper_weekly, donchian_lower_weekly = calculate_donchian(high_weekly, low_weekly, 20)
    
    # Align weekly Donchian to 12h timeframe
    donchian_upper_12h = align_htf_to_ltf(prices, df_weekly, donchian_upper_weekly)
    donchian_lower_12h = align_htf_to_ltf(prices, df_weekly, donchian_lower_weekly)
    
    # Get daily data for filters
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate daily EMA (50) for trend filter
    ema_50_daily = calculate_ema(close_daily, 50)
    ema_50_12h = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Calculate daily volume spike (volume > 1.5 * 20-day average)
    volume_ma_20_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    volume_spike_daily = volume_daily > (1.5 * volume_ma_20_daily)
    volume_spike_12h = align_htf_to_ltf(prices, df_daily, volume_spike_daily.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(volume_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian high + volume spike + above daily EMA
            if (close[i] > donchian_upper_12h[i] and 
                volume_spike_12h[i] > 0.5 and  # boolean as float
                close[i] > ema_50_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low + volume spike + below daily EMA
            elif (close[i] < donchian_lower_12h[i] and 
                  volume_spike_12h[i] > 0.5 and
                  close[i] < ema_50_12h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below weekly Donchian low
            if close[i] < donchian_lower_12h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above weekly Donchian high
            if close[i] > donchian_upper_12h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyDonchianBreakout_1dVolumeSpike_1dTrend"
timeframe = "12h"
leverage = 1.0