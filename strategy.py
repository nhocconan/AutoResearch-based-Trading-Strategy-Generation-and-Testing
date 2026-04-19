#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high_1d, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low_1d, 1)
    prev_low[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    # R4 = C + (H - L) * 1.1 / 2
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter