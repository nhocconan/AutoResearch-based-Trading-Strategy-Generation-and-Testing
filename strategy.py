#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout with 1-day Trend Filter and Volume Spike.
Long when price breaks above Camarilla H4 (resistance) during 1-day uptrend with volume spike.
Short when price breaks below Camarilla L4 (support) during 1-day downtrend with volume spike.
Exit when price returns to daily pivot or trend reverses.
Designed for low trade frequency by requiring trend alignment and volume confirmation.
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
    
    # Calculate Camarilla pivot levels from daily data
    # Use previous day's OHLC for today's levels (avoid look-ahead)
    daily_high = get_htf_data(prices, '1d')['high'].values
    daily_low = get_htf_data(prices, '1d')['low'].values
    daily_close = get_htf_data(prices, '1d')['close'].values
    
    # Calculate Camarilla levels: H4, L4, Pivot
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    camarilla_h4 = daily_close + 1.5 * (daily_high - daily_low)
    camarilla_l4 = daily_close - 1.5 * (daily_high - daily_low)
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), camarilla_l4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), camarilla_pivot)
    
    # 1-day trend filter: 20-period EMA
    daily_ema20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_ema20_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), daily_ema20)
    
    # Volume confirmation: current volume > 2.0x 20-period average (stricter for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(daily_ema20_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Camarilla H4 + 1-day uptrend + volume spike
            if close[i] > camarilla_h4_aligned[i] and daily_ema20_aligned[i] > daily_ema20_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L4 + 1-day downtrend + volume spike
            elif close[i] < camarilla_l4_aligned[i] and daily_ema20_aligned[i] < daily_ema20_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to daily pivot or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below pivot or 1-day trend turns down
                if close[i] < camarilla_pivot_aligned[i] or daily_ema20_aligned[i] < daily_ema20_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above pivot or 1-day trend turns up
                if close[i] > camarilla_pivot_aligned[i] or daily_ema20_aligned[i] > daily_ema20_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0