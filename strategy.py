#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R extremes with volume confirmation and 1d EMA50 trend filter.
# Enter long when 12h Williams %R < -80 (oversold) with volume > 1.5x 20-bar average and close > 1d EMA50.
# Enter short when 12h Williams %R > -20 (overbought) with volume > 1.5x average and close < 1d EMA50.
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is reached.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag.
# Williams %R identifies exhaustion points in trends, effective in both bull (buy pullbacks) and bear (sell rallies).
# 12h timeframe for Williams %R reduces noise vs lower TF, 1d EMA50 provides reliable trend filter.

name = "6h_WilliamsR_Extremes_12h_VolumeConfirm_1dEMA50_v1"
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
    
    # Get 12h data for Williams %R calculation (MTF oscillator)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R parameters
    williams_period = 14
    
    # Calculate highest high and lowest low over the period
    highest_high = pd.Series(high_12h).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=williams_period, min_periods=williams_period).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, ((highest_high - close_12h) / rr) * -100, -50)
    
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
    
    # Calculate volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
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
        
        # Williams %R conditions
        williams_r_val = williams_r_aligned[i]
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        mean_reversion_exit = abs(williams_r_val + 50) < 10  # Near -50
        opposite_extreme = (position == 1 and williams_r_val > -20) or (position == -1 and williams_r_val < -80)
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = oversold and vol_confirm and bullish_bias
        short_entry = overbought and vol_confirm and bearish_bias
        
        # Exit conditions
        long_exit = mean_reversion_exit or opposite_extreme
        short_exit = mean_reversion_exit or opposite_extreme
        
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