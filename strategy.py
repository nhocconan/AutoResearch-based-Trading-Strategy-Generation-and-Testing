#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with daily trend filter and volume confirmation.
Long when Williams %R crosses above -20 (oversold bounce) + daily close > daily EMA50 + volume > 1.5x average.
Short when Williams %R crosses below -80 (overbought rejection) + daily close < daily EMA50 + volume > 1.5x average.
Exit when Williams %R crosses -50 (mean reversion) or daily trend changes.
Designed for low trade frequency (~15-30/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(daily_ema50_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        daily_close_val = None
        daily_ema50_val = None
        if i < len(daily_ema50_aligned):
            daily_close_val = df_1d['close'].values[-1] if len(df_1d) > 0 else np.nan
            daily_ema50_val = daily_ema50_aligned[i]
        else:
            daily_close_val = np.nan
            daily_ema50_val = np.nan
            
        if np.isnan(daily_close_val) or np.isnan(daily_ema50_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_val > daily_ema50_val
        daily_trend_down = daily_close_val < daily_ema50_val
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -20 + daily uptrend + volume confirmation
            if (williams_r[i] > -20 and williams_r[i-1] <= -20 and 
                daily_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 + daily downtrend + volume confirmation
            elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and 
                  daily_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 or daily trend changes to down
                if williams_r[i] >= -50 or not daily_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 or daily trend changes to up
                if williams_r[i] <= -50 or not daily_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_DailyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0