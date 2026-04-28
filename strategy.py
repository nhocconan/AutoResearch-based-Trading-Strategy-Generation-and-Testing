#!/usr/bin/env python3
# Hypothesis: 1d price action within weekly Bollinger Bands with trend filter and volume confirmation.
# Uses weekly Bollinger Bands to identify overbought/oversold conditions relative to the weekly volatility envelope.
# Enters long when price touches lower band in weekly uptrend with volume confirmation, short when price touches upper band in weekly downtrend.
# Designed for 1d timeframe with ~30-100 total trades over 4 years to minimize fee decay.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    bb_period = 20
    bb_std = 2
    sma_1w = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev_1w = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band_1w = sma_1w + (bb_std_dev_1w * bb_std)
    lower_band_1w = sma_1w - (bb_std_dev_1w * bb_std)
    
    # Weekly trend: price above/below weekly SMA(50)
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    weekly_uptrend = close_1w > sma_50_1w
    weekly_downtrend = close_1w < sma_50_1w
    
    # Align weekly indicators to daily
    upper_band_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_band_1w)
    lower_band_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_band_1w)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 20)  # Wait for weekly BB, trend, and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_1w_aligned[i]) or np.isnan(lower_band_1w_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: touch Bollinger Band in trend direction with volume
        long_entry = (low[i] <= lower_band_1w_aligned[i]) and weekly_uptrend_aligned[i] and volume_confirm[i]
        short_entry = (high[i] >= upper_band_1w_aligned[i]) and weekly_downtrend_aligned[i] and volume_confirm[i]
        
        # Exit conditions: opposite band touch or loss of trend
        long_exit = (high[i] >= upper_band_1w_aligned[i]) or (weekly_uptrend_aligned[i] < 0.5)
        short_exit = (low[i] <= lower_band_1w_aligned[i]) or (weekly_downtrend_aligned[i] < 0.5)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "1d_WeeklyBollingerBand_Touch_Trend_Volume"
timeframe = "1d"
leverage = 1.0