#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with daily trend filter and volume confirmation.
Long when price > Alligator teeth (Jaw) + daily close > daily EMA50 + volume > 1.5x average volume.
Short when price < Alligator teeth (Jaw) + daily close < daily EMA50 + volume > 1.5x average volume.
Exit when price crosses Alligator lips (Lips) or daily trend changes.
Williams Alligator acts as a dynamic trend filter, reducing whipsaws in choppy markets.
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
    
    # Calculate Williams Alligator (13,8,5 smoothed with 8,5,3)
    # Jaw (13-period SMMA, smoothed 8)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean()
    
    # Teeth (8-period SMMA, smoothed 5)
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean()
    
    # Lips (5-period SMMA, smoothed 3)
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean()
    
    # Load daily data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(daily_ema50_aligned[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
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
            # Long: price > teeth (Alligator jaw) + daily uptrend + volume confirmation
            if (close[i] > teeth[i] and 
                daily_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price < teeth (Alligator jaw) + daily downtrend + volume confirmation
            elif (close[i] < teeth[i] and 
                  daily_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price < lips (Alligator lips) or daily trend changes to down
                if close[i] < lips[i] or not daily_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price > lips (Alligator lips) or daily trend changes to up
                if close[i] > lips[i] or not daily_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_DailyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0