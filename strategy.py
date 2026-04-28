#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R extremes with 1d EMA50 trend filter and volume confirmation.
# Enter long when 12h Williams %R < -80 (oversold) and price > 1d EMA50 with volume > 2x 20-bar average.
# Enter short when 12h Williams %R > -20 (overbought) and price < 1d EMA50 with volume > 2x average.
# Exit when Williams %R returns to -50 level.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Williams %R identifies exhaustion points; EMA50 filter ensures trades follow higher timeframe trend.
# Volume confirmation reduces false signals. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).

name = "6h_WilliamsR_Extremes_1dEMA50_VolumeConfirm_v1"
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
    
    # Get 12h data for Williams %R calculation (MTF structure)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 14-period Williams %R
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 1d data for EMA50 trend filter (MTF trend)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Williams %R conditions
        williams_r_val = williams_r_aligned[i]
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        exit_level = williams_r_val > -50 and williams_r_val < -50  # never true, will fix below
        
        # Exit condition: return to -50 level (mean reversion)
        long_exit = williams_r_val > -50
        short_exit = williams_r_val < -50
        
        # Entry conditions
        long_entry = oversold and vol_confirm and bullish_bias
        short_entry = overbought and vol_confirm and bearish_bias
        
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