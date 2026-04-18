#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) with 1-day EMA trend filter and volume confirmation.
Enters long when price breaks above R1 with EMA34 > EMA89 and volume spike, short when breaks below S1 with EMA34 < EMA89.
Camarilla levels provide precise intraday support/resistance, EMA filter ensures trend alignment, volume confirms momentum.
Designed for ~20-30 trades/year on 12h timeframe with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and EMAs
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Actually, standard Camarilla uses: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # But we'll use the more common: R1 = (high+low+close)/3 + 1.1*(high-low)/12
    # Simpler: pivot = (high+low+close)/3, then R1 = pivot + 1.1*(high-low)/12, S1 = pivot - 1.1*(high-low)/12
    
    # Calculate daily pivot and ranges
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    rng = daily_high - daily_low
    R1 = pivot + 1.1 * rng / 12.0
    S1 = pivot - 1.1 * rng / 12.0
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate daily EMAs for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False).mean().values
    ema89_1d = pd.Series(daily_close).ewm(span=89, adjust=False).mean().values
    
    # Align EMAs to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema89_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(ema89_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            if close[i] > R1_aligned[i] and ema34_aligned[i] > ema89_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with downtrend and volume spike
            elif close[i] < S1_aligned[i] and ema34_aligned[i] < ema89_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or trend weakens
            if close[i] < S1_aligned[i] or ema34_aligned[i] <= ema89_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or trend weakens
            if close[i] > R1_aligned[i] or ema34_aligned[i] >= ema89_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0