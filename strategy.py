#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with weekly trend filter and daily volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion.
# Weekly trend filter ensures we trade with the higher timeframe trend.
# Daily volume confirmation adds conviction to reversals.
# Designed for low trade frequency (12-37/year) in 12h timeframe to minimize fee drag.
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend).
name = "12h_WilliamsR_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14-period) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily average volume (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: price above/below EMA34
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume above daily average
        vol_confirm = volume[i] > vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND weekly uptrend AND volume confirmation
            if williams_r[i] < -80 and weekly_uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND weekly downtrend AND volume confirmation
            elif williams_r[i] > -20 and weekly_downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 (momentum shift) OR weekly trend turns down
            exit_condition = williams_r[i] > -50 or not weekly_uptrend
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 (momentum shift) OR weekly trend turns up
            exit_condition = williams_r[i] < -50 or not weekly_downtrend
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals