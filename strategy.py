#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14) for trend filter
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align daily pivot levels to daily timeframe
    daily_pivot_1d = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_1d = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_1d = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need weekly RSI, daily pivots, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(daily_pivot_1d[i]) or 
            np.isnan(daily_r1_1d[i]) or 
            np.isnan(daily_s1_1d[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Weekly RSI trend filter
        rsi_bullish = rsi_1w_aligned[i] > 50
        rsi_bearish = rsi_1w_aligned[i] < 50
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_1d[i]
        price_below_s1 = close[i] < daily_s1_1d[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume and weekly RSI bullish
            if (price_above_r1 and volume_filter and rsi_bullish):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume and weekly RSI bearish
            elif (price_below_s1 and volume_filter and rsi_bearish):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot
            if close[i] < daily_pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot
            if close[i] > daily_pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRSI_DailyPivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0