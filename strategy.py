#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: <-80 oversold, >-20 overbought
# In ranging markets: buy when %R crosses above -80 from below with volume confirmation
# In trending markets: sell when %R crosses below -20 from above with volume confirmation
# 1d EMA50 provides trend filter: only take longs when price > EMA50, shorts when price < EMA50
# Volume confirmation reduces false signals: require volume > 1.5 x 20-period EMA
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# This strategy works in both bull and bear markets by adapting to regime via EMA50 filter

name = "6h_WilliamsR_MeanReversion_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max and min for 14 periods
    high_max_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_min_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R formula
    williams_r = np.where(
        (high_max_14 - low_min_14) != 0,
        ((high_max_14 - close_1d) / (high_max_14 - low_min_14)) * -100,
        -50  # neutral when range is zero
    )
    
    # Align Williams %R to 6h timeframe (using completed 1d bar's values)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams %R mean reversion signals with 1d trend filter
        # Long: %R crosses above -80 from below (oversold bounce) + volume confirm + price > EMA50
        # Short: %R crosses below -20 from above (overbought rejection) + volume confirm + price < EMA50
        if position == 0:
            # Check for crossover: previous bar below -80, current bar at or above -80
            wr_cross_up = (i > 0 and williams_r_aligned[i-1] < -80 and williams_r_aligned[i] >= -80)
            # Check for crossover: previous bar above -20, current bar at or below -20
            wr_cross_down = (i > 0 and williams_r_aligned[i-1] > -20 and williams_r_aligned[i] <= -20)
            
            if wr_cross_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif wr_cross_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: %R crosses above -20 (overbought) OR price breaks below EMA50
            wr_exit_long = (i > 0 and williams_r_aligned[i-1] <= -20 and williams_r_aligned[i] > -20)
            if wr_exit_long or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: %R crosses below -80 (oversold) OR price breaks above EMA50
            wr_exit_short = (i > 0 and williams_r_aligned[i-1] >= -80 and williams_r_aligned[i] < -80)
            if wr_exit_short or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals