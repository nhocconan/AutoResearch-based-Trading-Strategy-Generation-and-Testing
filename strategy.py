#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_WeeklyTrend_Volume
Hypothesis: On 12h timeframe, breakouts above weekly R1 or below weekly S1 with volume confirmation and daily trend filter capture strong momentum moves. Weekly trend (via EMA50) filters direction, reducing false breakouts in sideways markets. Designed for low trade frequency (15-30/year) to minimize fee drag while capturing major trends in both bull and bear markets.
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
    
    # Get weekly data for EMA50 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close']
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Calculate weekly Camarilla levels (using weekly OHLC)
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = (high_1d - low_1d)
    r1_level = close_1d + (1.1 * camarilla_range) / 12
    s1_level = close_1d - (1.1 * camarilla_range) / 12
    
    # Align weekly Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume spike detection: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above weekly R1 with weekly uptrend and volume spike
            if price > r1 and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with weekly downtrend and volume spike
            elif price < s1 and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to weekly S1 or breaks below weekly EMA50
            if price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to weekly R1 or breaks above weekly EMA50
            if price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0