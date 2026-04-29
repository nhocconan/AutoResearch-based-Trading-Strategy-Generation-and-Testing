#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d Supertrend(10,3) trend filter and volume confirmation
# Long when price breaks above upper Donchian(20) AND 1d Supertrend is bullish AND volume > 1.5x 20-bar avg
# Short when price breaks below lower Donchian(20) AND 1d Supertrend is bearish AND volume > 1.5x 20-bar avg
# Exit when price crosses the opposite Donchian band (mean reversion within the channel)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid overtrading.
# Donchian channels provide structural support/resistance, Supertrend filters for trend alignment,
# volume confirmation reduces false breakouts. Works in both bull and bear regimes via trend filter.

name = "6h_Donchian20_Supertrend1d_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Supertrend (10,3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # ATR
    atr_period = 10
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3.0 * atr
    basic_lb = (high_1d + low_1d) / 2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(close_1d))
    final_lb = np.zeros(len(close_1d))
    supertrend = np.zeros(len(close_1d))
    trend = np.ones(len(close_1d))  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    supertrend[0] = final_ub[0]
    trend[0] = 1
    
    for i in range(1, len(close_1d)):
        # Final Upper Band
        if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Final Lower Band
        if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        # Supertrend and Trend
        if supertrend[i-1] == final_ub[i-1]:
            if close_1d[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
                trend[i] = -1
            else:
                supertrend[i] = final_lb[i]
                trend[i] = 1
        else:
            if close_1d[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
                trend[i] = 1
            else:
                supertrend[i] = final_ub[i]
                trend[i] = -1
    
    # Align Supertrend trend to 6h timeframe (1 = uptrend, -1 = downtrend)
    supertrend_trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_trend_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_trend = supertrend_trend_aligned[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian band (mean reversion)
            if curr_close < curr_lowest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian band (mean reversion)
            if curr_close > curr_highest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper Donchian AND 1d Supertrend is bullish AND volume confirmation
            if curr_close > curr_highest and curr_trend == 1 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian AND 1d Supertrend is bearish AND volume confirmation
            elif curr_close < curr_lowest and curr_trend == -1 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals