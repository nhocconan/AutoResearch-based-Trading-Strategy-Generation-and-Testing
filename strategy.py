#!/usr/bin/env python3
# Hypothesis: 12h 1-week Bollinger Band breakout with daily volume confirmation and 1-day trend filter
# Long when: price > upper Bollinger Band (1-week, 20-period, 2 std dev), volume > 1.5x daily average, daily EMA(50) rising
# Short when: price < lower Bollinger Band (1-week, 20-period, 2 std dev), volume > 1.5x daily average, daily EMA(50) falling
# Exit when: price crosses back inside Bollinger Bands OR daily trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown
# Target: 15-30 trades/year on 12h timeframe (60-120 total over 4 years)
# Designed to capture strong momentum moves with volatility confirmation in both bull and bear markets

name = "12h_BollingerBreakout_1wVolTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week Bollinger Bands (20-period, 2 std dev)
    close_1w = df_1w['close'].values
    bb_period = 20
    bb_std = 2
    
    # Calculate rolling mean and std
    bb_mean = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std().values
    
    # Upper and lower bands
    bb_upper = bb_mean + (bb_std * bb_std_dev)
    bb_lower = bb_mean - (bb_std * bb_std_dev)
    
    # Align weekly Bollinger Bands to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Get daily data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising_1d = ema_50_1d > ema_50_1d_prev
    ema_falling_1d = ema_50_1d < ema_50_1d_prev
    
    # Align daily indicators to 12h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising_1d)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > upper Bollinger Band + volume spike + daily EMA rising
            if (close[i] > bb_upper_aligned[i] and 
                vol_spike_aligned[i] and 
                ema_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < lower Bollinger Band + volume spike + daily EMA falling
            elif (close[i] < bb_lower_aligned[i] and 
                  vol_spike_aligned[i] and 
                  ema_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below upper Bollinger Band OR daily trend turns down
            if (close[i] < bb_upper_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above lower Bollinger Band OR daily trend turns up
            if (close[i] > bb_lower_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals