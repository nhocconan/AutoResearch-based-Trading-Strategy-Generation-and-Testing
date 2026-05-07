#!/usr/bin/env python3
name = "1d_1w_Donchian_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly levels to daily
    upper_band = align_htf_to_ltf(prices, df_1w, high_20)
    lower_band = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Weekly trend: EMA(34) on weekly close
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection: 10-day average
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 10)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly upper band with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_10[i] * 2.0
            uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > upper_band[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower band with volume and weekly downtrend
            elif close[i] < lower_band[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below weekly upper band or volume drops
            if close[i] < upper_band[i] or volume[i] < vol_ma_10[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above weekly lower band or volume drops
            if close[i] > lower_band[i] or volume[i] < vol_ma_10[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian breakout with weekly trend and volume confirmation
# - Weekly Donchian channels provide strong support/resistance levels
# - Breakout above weekly upper band with volume in weekly uptrend = long opportunity
# - Breakdown below weekly lower band with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Weekly EMA(34) filter ensures trading with the higher timeframe trend
# - Position size 0.25 targets ~15-25 trades/year, minimizing fee drag
# - Works in both bull (buy upper band breaks in uptrend) and bear (sell lower band breaks in downtrend)
# - Exit when price returns to weekly band or volume weakens
# - Weekly timeframe avoids noise and false breakouts on daily chart
# - Donchian channels adapt to volatility, providing dynamic support/resistance levels