#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with weekly trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. Weekly trend filter ensures we trade
# only in the direction of the higher timeframe trend. Volume confirmation adds conviction.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend).
name = "6h_WilliamsR_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams %R (14-period) - using current bar's high/low/close
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 6h timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(weekly_ema_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Trend filter: price above/below weekly EMA
        price_above_weekly_ema = close[i] > weekly_ema_aligned[i]
        price_below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above weekly EMA (uptrend) AND volume confirmation
            long_signal = williams_r[i] < -80
            if vol_confirm and long_signal and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below weekly EMA (downtrend) AND volume confirmation
            elif vol_confirm and williams_r[i] > -20 and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) OR trend changes
            exit_condition = williams_r[i] > -50 or close[i] < weekly_ema_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) OR trend changes
            exit_condition = williams_r[i] < -50 or close[i] > weekly_ema_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals