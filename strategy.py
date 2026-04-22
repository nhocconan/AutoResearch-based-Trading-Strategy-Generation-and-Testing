#!/usr/bin/env python3

"""
Hypothesis: Daily Camarilla Pivot R4/S4 breakout with volume confirmation and 1-day EMA trend filter.
Goes long when price breaks above R4 during bullish trend (price > EMA34) with volume spike,
short when breaks below S4 during bearish trend (price < EMA34) with volume spike.
Exits on opposite pivot touch (S4 for longs, R4 for shorts). Designed for low trade frequency
(12-37/year) by requiring breakout of extreme pivot levels, trend alignment, and volume confirmation.
Works in both bull and bear markets by following daily trend via EMA34.
"""

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
    
    # Load daily data for Camarilla pivots and EMA34 - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Formula: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    daily_close = df_daily['close'].values
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    camarilla_r4 = daily_close + ((daily_high - daily_low) * 1.1 / 2)
    camarilla_s4 = daily_close - ((daily_high - daily_low) * 1.1 / 2)
    
    # Align to 12h timeframe (these levels are valid for the entire day after daily close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R4 + bullish trend (price > EMA34) + volume spike
            if close[i] > camarilla_r4_aligned[i] and close[i] > ema34_daily_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + bearish trend (price < EMA34) + volume spike
            elif close[i] < camarilla_s4_aligned[i] and close[i] < ema34_daily_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price touches opposite pivot level
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches or goes below S4
                if close[i] <= camarilla_s4_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches or goes above R4
                if close[i] >= camarilla_r4_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Camarilla_R4S4_Breakout_EMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0