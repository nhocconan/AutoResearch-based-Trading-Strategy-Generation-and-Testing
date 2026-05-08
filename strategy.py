#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly trend filter + volume confirmation
# Long when price breaks above Donchian upper band, price > weekly EMA50, volume > 1.5x average
# Short when price breaks below Donchian lower band, price < weekly EMA50, volume > 1.5x average
# Donchian channels capture breakouts, weekly EMA filters trend direction, volume confirms momentum
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_Donchian20_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Donchian channels (20-period) on 6h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend = ema50_weekly_aligned[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        vol_conf = vol_confirm[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: break above upper band, weekly uptrend, volume confirmation
            if close_val > upper_band and close_val > weekly_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band, weekly downtrend, volume confirmation
            elif close_val < lower_band and close_val < weekly_trend and vol_conf:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below lower band or weekly trend turns down
            if close_val < lower_band or close_val < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above upper band or weekly trend turns up
            if close_val > upper_band or close_val > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals