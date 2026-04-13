#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter and volume confirmation.
# Uses weekly Williams %R for overbought/oversold conditions and weekly trend direction (price vs 50-week SMA).
# Enters on mean reversion from extreme levels when aligned with weekly trend.
# Volume confirmation ensures rejection of extreme levels has conviction.
# Timeframe 6h reduces trade frequency to minimize fee drag while capturing swings.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    williams_r_values = williams_r
    
    # Calculate 50-week SMA for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Calculate weekly volume and its 20-period average
    volume_1w = df_1w['volume'].values
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 6-hour timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r_values)
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(sma_50_aligned[i]) or
            np.isnan(volume_ma_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x weekly volume MA (adjusted for 6h)
        # ~28 6h periods per week, so weekly MA/28 = approximate 6h period MA
        volume_6h_approx_ma = volume_ma_20_1w_aligned[i] / 28
        volume_condition = volume[i] > (volume_6h_approx_ma * 1.5)
        
        # Williams %R conditions: extreme oversold/overbought
        williams_r_oversold = williams_r_aligned[i] <= -80
        williams_r_overbought = williams_r_aligned[i] >= -20
        
        # Weekly trend filter: price above/below 50-week SMA
        weekly_uptrend = close_1w_aligned[i] > sma_50_aligned[i]
        weekly_downtrend = close_1w_aligned[i] < sma_50_aligned[i]
        
        # Entry conditions: Williams %R mean reversion with trend and volume
        # Long when Williams %R rebounds from oversold in uptrend with volume
        # Short when Williams %R rejects from overbought in downtrend with volume
        if position == 0:
            long_condition = williams_r_oversold and weekly_uptrend and volume_condition
            short_condition = williams_r_overbought and weekly_downtrend and volume_condition
            
            if long_condition:
                position = 1
                signals[i] = position_size
            elif short_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when Williams %R reaches overbought or trend changes
            if williams_r_aligned[i] >= -20 or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when Williams %R reaches oversold or trend changes
            if williams_r_aligned[i] <= -80 or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_WilliamsR_MeanReversion_Trend_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0