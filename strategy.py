#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d/1w trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + 1d EMA(50) uptrend + 1w EMA(100) uptrend + volume spike
# Short when price breaks below Donchian(20) low + 1d EMA(50) downtrend + 1w EMA(100) downtrend + volume spike
# Uses price channel breakouts for trend capture with multi-timeframe trend alignment
# Volume spike confirms institutional participation
# Targets 20-50 trades/year to minimize fee drag while capturing major trends

name = "4h_Donchian20_1d1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data once for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate weekly EMA(100) for trend filter
    weekly_close = df_1w['close'].values
    ema100_1w = pd.Series(weekly_close).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # Calculate Donchian(20) channels (4h timeframe)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema100_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        ema100_1w_val = ema100_1w_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Donchian breakout up + daily uptrend + weekly uptrend + volume spike
            if (close[i] > donch_high and 
                close[i] > ema50_1d_val and 
                close[i] > ema100_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakout down + daily downtrend + weekly downtrend + volume spike
            elif (close[i] < donch_low and 
                  close[i] < ema50_1d_val and 
                  close[i] < ema100_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakout down OR trend turns down
            if close[i] < donch_low or close[i] < ema50_1d_val or close[i] < ema100_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout up OR trend turns up
            if close[i] > donch_high or close[i] > ema50_1d_val or close[i] > ema100_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals