#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1d EMA34 trend filter and volume confirmation.
# Enter long when 1d Williams %R < -80 (oversold) with volume > 1.8x 50-bar average and close > 1d EMA34.
# Enter short when 1d Williams %R > -20 (overbought) with volume > 1.8x average and close < 1d EMA34.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Williams %R is effective in ranging markets (2025-2026 bear/range) and catches reversals in trends.
# The 1d EMA34 filter ensures trades align with higher-timeframe trend, reducing whipsaws.
# Volume confirmation ensures breakouts have conviction.

name = "6h_WilliamsR_Extremes_1dEMA34_VolumeConfirm_v1"
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
    
    # Get 1d data for Williams %R and EMA34 (MTF structure and trend)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need at least 34 days for EMA34
        return np.zeros(n)
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: >1.8x 50-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_50 = volume_series.rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > 1.8 * volume_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Williams %R conditions
        wr = williams_r_aligned[i]
        wr_oversold = wr < -80
        wr_overbought = wr > -20
        wr_exit_long = wr > -50  # Exit long when WR crosses above -50
        wr_exit_short = wr < -50  # Exit short when WR crosses below -50
        
        # Trend filter: 1d EMA34 bias
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions
        long_entry = wr_oversold and vol_confirm and bullish_bias
        short_entry = wr_overbought and vol_confirm and bearish_bias
        
        # Exit conditions
        long_exit = wr_exit_long
        short_exit = wr_exit_short
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals