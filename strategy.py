#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla pivot breakout with 1-day trend filter and volume confirmation.
Trade long when price breaks above Camarilla R1 level during 1-day uptrend with volume spike.
Trade short when price breaks below Camarilla S1 level during 1-day downtrend with volume spike.
Exit when price returns to the Camarilla Pivot point or trend reverses.
Designed for moderate trade frequency (20-50 trades/year) by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 1-day trend.
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
    
    # Calculate Camarilla pivot levels (based on previous day's data)
    # We'll use 1-day data to calculate pivot levels for current period
    pivot_high = high
    pivot_low = low
    pivot_close = close
    
    # Calculate typical price for pivot calculation
    typical_price = (pivot_high + pivot_low + pivot_close) / 3
    
    # Calculate Camarilla levels using typical price range
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    # Pivot = (High + Low + Close) / 3
    hl_range = pivot_high - pivot_low
    camarilla_r1 = pivot_close + hl_range * 1.1 / 12
    camarilla_s1 = pivot_close - hl_range * 1.1 / 12
    camarilla_pivot = (pivot_high + pivot_low + pivot_close) / 3
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + 1d uptrend + volume spike
            if close[i] > camarilla_r1[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + 1d downtrend + volume spike
            elif close[i] < camarilla_s1[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to pivot or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below pivot or 1d trend turns down
                if close[i] < camarilla_pivot[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above pivot or 1d trend turns up
                if close[i] > camarilla_pivot[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0