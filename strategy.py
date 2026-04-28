#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Breakout_DailyTrend_VolumeSpike
Hypothesis: Uses weekly R3/S3 levels from the 1-week timeframe for breakout trades.
Trades breakouts above weekly R3 in uptrend (daily EMA34 up) or below weekly S3 in downtrend (daily EMA34 down).
Confirmed by volume spikes (>2x 20-period MA). Uses 6h timeframe to capture intermediate-term moves.
Designed to work in bull markets (breakouts above R3) and bear markets (breakdowns below S3).
Targets 12-37 trades per year to minimize fee drag while capturing significant momentum moves.
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
    
    # Get 1-week data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot points (standard floor pivot)
    # Using previous week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    R3 = pivot + 2 * (weekly_high - weekly_low)
    S3 = pivot - 2 * (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Calculate volume spike (>2x 20-period MA for strong confirmation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1-day EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Price relative to weekly R3/S3
        price_above_R3 = close[i] > R3_aligned[i]
        price_below_S3 = close[i] < S3_aligned[i]
        
        # Entry logic:
        # Long: Breakout above weekly R3 in uptrend with volume
        long_entry = vol_confirm and trend_up and price_above_R3 and (close[i-1] <= R3_aligned[i-1])
        
        # Short: Breakdown below weekly S3 in downtrend with volume
        short_entry = vol_confirm and trend_down and price_below_S3 and (close[i-1] >= S3_aligned[i-1])
        
        # Exit logic: Opposite level or trend reversal
        long_exit = (price_below_S3 and close[i] > S3_aligned[i-1]) or not trend_up
        short_exit = (price_above_R3 and close[i] < R3_aligned[i-1]) or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_DailyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0