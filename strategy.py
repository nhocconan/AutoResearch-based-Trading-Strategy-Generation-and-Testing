#!/usr/bin/env python3
"""
Hypothesis: 1-day CCI with weekly trend filter and volume confirmation.
Long when CCI crosses above -100 (mean reversion) + weekly close > weekly EMA50 + volume > 1.5x average.
Short when CCI crosses below +100 (mean reversion) + weekly close < weekly EMA50 + volume > 1.5x average.
Exit when CCI crosses zero (mean reversion complete) or weekly trend changes.
Designed for low trade frequency (~10-20/year) to minimize fee drift in both bull and bear markets.
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
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Calculate CCI (20-period)
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - sma_tp.values) / (0.015 * mad.values)
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(cci[i]) or np.isnan(weekly_ema50_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_close_val = None
        weekly_ema50_val = None
        if i < len(weekly_ema50_aligned):
            weekly_close_val = df_1w['close'].values[-1] if len(df_1w) > 0 else np.nan
            weekly_ema50_val = weekly_ema50_aligned[i]
        else:
            weekly_close_val = np.nan
            weekly_ema50_val = np.nan
            
        if np.isnan(weekly_close_val) or np.isnan(weekly_ema50_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_trend_up = weekly_close_val > weekly_ema50_val
        weekly_trend_down = weekly_close_val < weekly_ema50_val
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: CCI crosses above -100 + weekly uptrend + volume confirmation
            if (cci[i] > -100 and cci[i-1] <= -100 and 
                weekly_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses below +100 + weekly downtrend + volume confirmation
            elif (cci[i] < 100 and cci[i-1] >= 100 and 
                  weekly_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: CCI crosses above +100 or weekly trend changes to down
                if cci[i] >= 100 or not weekly_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: CCI crosses below -100 or weekly trend changes to up
                if cci[i] <= -100 or not weekly_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_CCI_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0