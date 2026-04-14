#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1w Supertrend for trend direction and 12h Donchian breakout for entry.
# 1w Supertrend (ATR=10, multiplier=3) filters for strong trends to avoid whipsaws in ranging markets.
# 12h Donchian channel (period=20) provides breakout signals in the direction of the weekly trend.
# Volume confirmation (>1.3x 20-period average) reduces false breakouts.
# ATR-based trailing stop manages risk by exiting when price moves against position by 2.5x ATR.
# Designed to work in both bull and bear markets by using 1w trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Supertrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR
    atr_period = 10
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2 + 3 * atr
    basic_lb = (high_1w + low_1w) / 2 - 3 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(close_1w))
    final_lb = np.zeros(len(close_1w))
    
    for i in range(len(close_1w)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(len(close_1w))
    for i in range(len(close_1w)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close_1w[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
            else:
                if close_1w[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                else:
                    supertrend[i] = final_ub[i]
    
    # Trend direction: 1 for uptrend, -1 for downtrend
    trend = np.where(close_1w > supertrend, 1, -1)
    
    # Load 12h data ONCE for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_period = 20
    upper_channel = pd.Series(high_12h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_12h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align indicators to 12h timeframe
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for trailing stop (using 12h data)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    tr1_12h = np.abs(high_12h[1:] - low_12h[1:])
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    
    atr_period_12h = 10
    atr_12h = pd.Series(tr_12h).ewm(span=atr_period_12h, adjust=False, min_periods=atr_period_12h).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donchian_period, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts in direction of weekly trend
            # Long: price breaks above upper Donchian channel AND weekly uptrend
            if (close[i] > upper_channel_aligned[i] and 
                trend_aligned[i] == 1 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian channel AND weekly downtrend
            elif (close[i] < lower_channel_aligned[i] and 
                  trend_aligned[i] == -1 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price drops below lower Donchian channel or trailing stop hit
            # Trailing stop: exit if price drops by 2.5x ATR from highest high since entry
            if (close[i] < lower_channel_aligned[i] or 
                close[i] <= highest_high - 2.5 * atr_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                # Update highest high for trailing stop
                if 'highest_high' not in locals():
                    highest_high = close[i]
                else:
                    highest_high = max(highest_high, close[i])
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above upper Donchian channel or trailing stop hit
            # Trailing stop: exit if price rises by 2.5x ATR from lowest low since entry
            if (close[i] > upper_channel_aligned[i] or 
                close[i] >= lowest_low + 2.5 * atr_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                # Update lowest low for trailing stop
                if 'lowest_low' not in locals():
                    lowest_low = close[i]
                else:
                    lowest_low = min(lowest_low, close[i])
                signals[i] = -position_size
    
    return signals

name = "12h_1wSupertrend_12hDonchian_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0