#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike
Hypothesis: On 4-hour timeframe, use Camarilla R1/S1 breakouts in the direction of daily EMA34 trend with volume confirmation. 
Camarilla levels provide precise support/resistance from daily action, EMA34 filters trend direction, and volume surge confirms institutional participation.
Designed for 20-50 trades/year to avoid fee drag while capturing meaningful moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and EMA trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We focus on R1 and S1 (inner levels)
    daily_close = df_daily['close'].values
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Calculate Camarilla R1 and S1 for each day
    # R1 = C + ((H-L)*1.1/6)
    # S1 = C - ((H-L)*1.1/6)
    camarilla_r1 = daily_close + ((daily_high - daily_low) * 1.1 / 6)
    camarilla_s1 = daily_close - ((daily_high - daily_low) * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Daily trend: bullish when price > EMA34
    daily_uptrend = df_daily['close'].values > ema34_daily
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_daily, daily_uptrend)
    daily_downtrend_aligned = ~daily_uptrend_aligned
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema34_daily_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price breaks Camarilla R1/S1 with trend and volume
        long_entry = (close[i] > camarilla_r1_aligned[i]) and daily_uptrend_aligned[i] and volume_surge[i]
        short_entry = (close[i] < camarilla_s1_aligned[i]) and daily_downtrend_aligned[i] and volume_surge[i]
        
        # Exit conditions: reverse signal with volume surge
        long_exit = (close[i] < camarilla_s1_aligned[i]) and volume_surge[i]
        short_exit = (close[i] > camarilla_r1_aligned[i]) and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0