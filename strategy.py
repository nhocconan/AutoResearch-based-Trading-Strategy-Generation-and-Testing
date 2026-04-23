#!/usr/bin/env python3
"""
Hypothesis: 1-day RSI with weekly Bollinger Band squeeze and volume confirmation.
Long when RSI crosses above 30 (mean reversion) + weekly Bollinger Band width < 20th percentile + volume > 1.5x average.
Short when RSI crosses below 70 (mean reversion) + weekly Bollinger Band width < 20th percentile + volume > 1.5x average.
Exit when RSI crosses 50 (mean reversion complete) or weekly Bollinger Band width > 80th percentile (volatility expansion).
Designed for low trade frequency (~10-20/year) to minimize fee drift in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for Bollinger Band squeeze filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20-period, 2 std)
    weekly_close = df_1w['close'].values
    weekly_sma20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean()
    weekly_std20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).std()
    weekly_upper = weekly_sma20 + 2 * weekly_std20
    weekly_lower = weekly_sma20 - 2 * weekly_std20
    weekly_bb_width = (weekly_upper - weekly_lower) / weekly_sma20 * 100  # as percentage
    
    # Calculate percentile ranks for BB width (20th and 80th percentiles)
    weekly_bb_width_series = pd.Series(weekly_bb_width)
    weekly_bb_width_20th = weekly_bb_width_series.rolling(window=50, min_periods=50).quantile(0.20)
    weekly_bb_width_80th = weekly_bb_width_series.rolling(window=50, min_periods=50).quantile(0.80)
    weekly_bb_width_20th_vals = weekly_bb_width_20th.values
    weekly_bb_width_80th_vals = weekly_bb_width_80th.values
    
    # Align BB width percentiles to daily timeframe
    weekly_bb_width_20th_aligned = align_htf_to_ltf(prices, df_1w, weekly_bb_width_20th_vals)
    weekly_bb_width_80th_aligned = align_htf_to_ltf(prices, df_1w, weekly_bb_width_80th_vals)
    
    # Calculate daily RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_vals = rsi.values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(rsi_vals[i]) or np.isnan(weekly_bb_width_20th_aligned[i]) or 
            np.isnan(weekly_bb_width_80th_aligned[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_squeeze = weekly_bb_width_20th_aligned[i] > 0 and weekly_bb_width_80th_aligned[i] > 0
        if not bb_squeeze:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        vol_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: RSI crosses above 30 + BB squeeze + volume confirmation
            if (rsi_vals[i] > 30 and rsi_vals[i-1] <= 30 and 
                weekly_bb_width_80th_aligned[i] > weekly_bb_width[i] and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 70 + BB squeeze + volume confirmation
            elif (rsi_vals[i] < 70 and rsi_vals[i-1] >= 70 and 
                  weekly_bb_width_80th_aligned[i] > weekly_bb_width[i] and vol_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses above 50 or BB width expands beyond 80th percentile
                if rsi_vals[i] >= 50 or weekly_bb_width[i] >= weekly_bb_width_80th_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses below 50 or BB width expands beyond 80th percentile
                if rsi_vals[i] <= 50 or weekly_bb_width[i] >= weekly_bb_width_80th_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_RSI_WeeklyBBSqueeze_VolumeFilter"
timeframe = "1d"
leverage = 1.0