#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h primary timeframe with 1d/1w HTF filters
    # Long: price breaks above 1d Donchian(20) high + 1w EMA50 > EMA200 (bull trend) + volume > 1.5x 20-period avg
    # Short: price breaks below 1d Donchian(20) low + 1w EMA50 < EMA200 (bear trend) + volume > 1.5x 20-period avg
    # Exit: price returns to 1d Donchian middle
    # Target: 75-200 total trades over 4 years (19-50/year) to balance signal quality and fee drag
    # Uses weekly trend filter to avoid counter-trend trades in both bull/bear markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for primary timeframe (Donchian channels and volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 1w data for trend filter (EMA crossover)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels on 1d data (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate EMAs on 1w data for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    weekly_bull = ema_50_1w > ema_200_1w  # Bullish weekly trend
    weekly_bear = ema_50_1w < ema_200_1w  # Bearish weekly trend
    
    # Volume average on 1d data (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_bull)
    weekly_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_bear)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(weekly_bull_aligned[i]) or 
            np.isnan(weekly_bear_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average (aligned from 1d)
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Entry conditions: breakout + volume + weekly trend alignment
        enter_long = breakout_up and volume_confirmed and weekly_bull_aligned[i]
        enter_short = breakout_down and volume_confirmed and weekly_bear_aligned[i]
        
        # Exit conditions: price returns to 1d Donchian middle
        exit_long = position == 1 and close[i] <= donchian_mid_aligned[i]
        exit_short = position == -1 and close[i] >= donchian_mid_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0