#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal with Volume Confirmation and 1d EMA200 Trend Filter
# Weekly pivots from prior week provide strong support/resistance levels.
# Price rejection at weekly R1/S1 with volume confirmation indicates institutional defense of levels.
# 1d EMA200 filter ensures alignment with long-term trend to avoid counter-trend trades.
# Works in bull markets (bounces off weekly S1) and bear markets (rejections at weekly R1).
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
name = "6h_WeeklyPivot_R1S1_Reversal_Volume_EMA200"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels from previous week's OHLC
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot and support/resistance levels (using previous week)
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3
    range_weekly = high_weekly - low_weekly
    
    # Weekly R1 and S1 levels
    r1_weekly = close_weekly + (range_weekly * 1.1 / 12)
    s1_weekly = close_weekly - (range_weekly * 1.1 / 12)
    
    # Align weekly pivot levels to 6h timeframe (using previous week's values)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Get daily data for EMA200 trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Calculate EMA200 on daily data
    ema_200_daily = pd.Series(close_daily).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_200_daily)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume (5 days on 6h chart)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or
            np.isnan(ema_200_daily_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_weekly_val = r1_weekly_aligned[i]
        s1_weekly_val = s1_weekly_aligned[i]
        ema_val = ema_200_daily_aligned[i]
        
        if position == 0:
            # Long: Price bounces off weekly S1 with volume confirmation and above EMA200
            if close_val > s1_weekly_val and low[i] <= s1_weekly_val and volume_confirm[i] and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: Price rejects at weekly R1 with volume confirmation and below EMA200
            elif close_val < r1_weekly_val and high[i] >= r1_weekly_val and volume_confirm[i] and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below weekly S1 or reaches weekly R1
            if close_val < s1_weekly_val or close_val >= r1_weekly_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above weekly R1 or reaches weekly S1
            if close_val > r1_weekly_val or close_val <= s1_weekly_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals