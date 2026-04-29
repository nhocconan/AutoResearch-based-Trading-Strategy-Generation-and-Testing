#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above 6h Donchian upper band with volume spike AND weekly pivot shows bullish bias
# Short when price breaks below 6h Donchian lower band with volume spike AND weekly pivot shows bearish bias
# Weekly pivot direction provides higher-timeframe structure to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing strong breakouts
# Uses discrete position sizing (0.25) to reduce churn and manage drawdown

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points and direction
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Bullish bias: weekly_close > weekly_pivot
    # Bearish bias: weekly_close < weekly_pivot
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    weekly_bullish = close_1w > pivot_1w  # Bullish bias when close above pivot
    weekly_bearish = close_1w < pivot_1w  # Bearish bias when close below pivot
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_weekly_bullish = weekly_bullish_aligned[i] > 0.5
        curr_weekly_bearish = weekly_bearish_aligned[i] > 0.5
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation to avoid false breakouts
            if curr_volume_confirm:
                # Bullish entry: price breaks above Donchian upper band with volume AND weekly bullish bias
                if curr_high > curr_dc_upper and curr_weekly_bullish:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower band with volume AND weekly bearish bias
                elif curr_low < curr_dc_lower and curr_weekly_bearish:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when price falls below Donchian lower band or weekly bias turns bearish
            if curr_low < curr_dc_lower or not curr_weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price rises above Donchian upper band or weekly bias turns bullish
            if curr_high > curr_dc_upper or not curr_weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals