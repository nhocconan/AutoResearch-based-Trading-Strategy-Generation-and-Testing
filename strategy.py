#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation.
Only trade long when price breaks above Donchian(20) upper band with 1-day EMA34 up and volume spike;
short when price breaks below Donchian(20) lower band with 1-day EMA34 down and volume spike.
Uses Donchian channel breakouts as momentum signals, filtered by daily trend and volume to avoid false breakouts.
Designed for low trade frequency (19-50 trades/year) by requiring multiple confirmations: breakout, trend alignment, and volume spike.
Works in both bull and bear markets by following the daily trend direction.
"""

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
    
    # Donchian Channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Daily EMA34 for trend direction
    daily_close = df_daily['close'].values
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: breakout above Donchian upper + daily uptrend + volume spike
            if close[i] > donchian_high[i] and ema34_daily_aligned[i] > ema34_daily_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian lower + daily downtrend + volume spike
            elif close[i] < donchian_low[i] and ema34_daily_aligned[i] < ema34_daily_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of Donchian channel or trend reverses
            exit_signal = False
            
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            
            if position == 1:
                # Exit long: price returns to middle or daily trend turns down
                if close[i] < donchian_mid or ema34_daily_aligned[i] < ema34_daily_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle or daily trend turns up
                if close[i] > donchian_mid or ema34_daily_aligned[i] > ema34_daily_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0