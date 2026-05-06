#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA50 trend filter and session filter (08-20 UTC)
# Long when price breaks above 4h Donchian upper channel AND 1d EMA50 is rising AND in session
# Short when price breaks below 4h Donchian lower channel AND 1d EMA50 is falling AND in session
# Exit when price crosses the 4h Donchian middle (mean) OR exits session
# Uses discrete sizing 0.20 to control drawdown and fee drag
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Donchian channel provides clear trend structure and breakout signals
# 1d EMA50 ensures we trade with the daily trend while reducing noise
# Session filter (08-20 UTC) avoids low-liquidity periods and reduces noise trades
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "1h_4hDonchian20_Breakout_1dEMA50_Trend_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data ONCE before loop for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars for Donchian(20)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channel (20-period)
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_high_20 + lowest_low_20) / 2.0
    
    # Align 4h Donchian channels to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, highest_high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC (already datetime64[ms] in prices.index)
    hours = prices.index.hour  # Pre-compute once before loop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        in_session = 8 <= hours[i] <= 20
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper channel, EMA50 rising, in session
            if (close[i] > donchian_upper_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower channel, EMA50 falling, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h Donchian middle OR exits session
            if close[i] < donchian_middle_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h Donchian middle OR exits session
            if close[i] > donchian_middle_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals