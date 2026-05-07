#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_TrendVolume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-week lookback)
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly levels to daily timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Weekly trend filter: EMA(13) on weekly close
    ema_13_1w = pd.Series(df_1w['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Volume spike detection: 3-day average (to avoid noise)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 3)  # Wait for Donchian, EMA, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or 
            np.isnan(ema_13_1w_aligned[i]) or np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            weekly_uptrend = ema_13_1w_aligned[i] > ema_13_1w_aligned[i-1]
            
            if close[i] > high_20w_aligned[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and weekly downtrend
            elif close[i] < low_20w_aligned[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below weekly Donchian low or volume drops
            if close[i] < low_20w_aligned[i] or volume[i] < vol_ma_3[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above weekly Donchian high or volume drops
            if close[i] > high_20w_aligned[i] or volume[i] < vol_ma_3[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with volume confirmation and weekly trend filter
# - Weekly Donchian(20) captures major support/resistance from institutional timeframe
# - Breakout above weekly high with 2x volume in weekly uptrend = high-probability long
# - Breakdown below weekly low with 2x volume in weekly downtrend = high-probability short
# - Volume confirmation (2x 3-day average) filters false breakouts
# - Weekly EMA(13) trend filter ensures trades align with higher timeframe momentum
# - Designed for low trade frequency: targets 15-30 trades/year to minimize fee drag
# - Works in both bull (buy weekly high breaks) and bear (sell weekly low breaks)
# - Position size 0.25 balances return potential with drawdown control
# - Uses actual weekly data from Binance (no resampling) via mtf_data
# - Avoids overtrading by requiring multiple confluence factors for entry