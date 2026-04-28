#!/usr/bin/env python3
# 1d_1w_Camarilla_Pivot_With_WeeklyTrend_Volume
# Hypothesis: Use weekly Camarilla pivot levels (R1/S1) on 1d timeframe with weekly trend filter and volume confirmation.
# Captures reversions to weekly mean in ranging markets and breakouts in trending markets.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume surge confirms institutional participation. Targets 7-25 trades/year to minimize fee drag.

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
    
    # Calculate weekly Camarilla pivot levels (R1/S1) on daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Use weekly high, low, close from previous week
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().values  # 5 days = 1 week
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().values
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().values
    
    # Calculate Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_r1 = weekly_close + (weekly_high - weekly_low) * 1.1 / 12
    camarilla_s1 = weekly_close - (weekly_high - weekly_low) * 1.1 / 12
    
    # Get weekly EMA50 for trend filter (using weekly close)
    ema_50_weekly = pd.Series(weekly_close).ewm(span=10, adjust=False, min_periods=10).mean().values  # 10 weeks ~ 2 months
    
    # Align higher timeframe data to daily
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_1d, ema_50_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_weekly_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to Camarilla levels
        price_near_r1 = abs(close[i] - camarilla_r1_aligned[i]) / camarilla_r1_aligned[i] < 0.005  # Within 0.5%
        price_near_s1 = abs(close[i] - camarilla_s1_aligned[i]) / camarilla_s1_aligned[i] < 0.005  # Within 0.5%
        
        # Trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema_50_weekly_aligned[i]
        trend_down = close[i] < ema_50_weekly_aligned[i]
        
        # Entry conditions
        # Long: price near S1 support + uptrend + volume surge
        long_entry = price_near_s1 and trend_up and volume_surge[i]
        # Short: price near R1 resistance + downtrend + volume surge
        short_entry = price_near_r1 and trend_down and volume_surge[i]
        
        # Exit conditions: price moves away from level or trend reversal
        long_exit = close[i] > camarilla_r1_aligned[i] or not trend_up
        short_exit = close[i] < camarilla_s1_aligned[i] or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_Camarilla_Pivot_With_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0