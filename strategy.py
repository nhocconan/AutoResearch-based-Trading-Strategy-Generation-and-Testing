#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike
- Long: Price breaks above Camarilla R1 + price > 12h EMA50 + volume > 2x 20-period average
- Short: Price breaks below Camarilla S1 + price < 12h EMA50 + volume > 2x 20-period average
- Exit: Price retests the Camarilla pivot level (mean reversion to avoid whipsaws)
- Uses Camarilla levels from 1d timeframe for institutional reference points
- Volume confirmation ensures institutional participation
- 12h EMA50 filter ensures we trade with the intermediate-term trend
- Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
- Works in both bull and bear markets by trading breakouts with institutional level filters
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
    
    # Get 1d data for Camarilla pivot levels (from prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day
    # Camarilla levels: based on prior day's range
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    prior_high = df_1d['high'].shift(1).values  # Prior day high
    prior_low = df_1d['low'].shift(1).values    # Prior day low
    prior_close = df_1d['close'].shift(1).values # Prior day close
    
    # Calculate pivot and Camarilla levels
    pp = (prior_high + prior_low + prior_close) / 3.0
    rng = prior_high - prior_low
    
    r1 = prior_close + (rng * 1.1 / 12)  # Resistance 1
    s1 = prior_close - (rng * 1.1 / 12)  # Support 1
    pivot = pp  # Pivot point for exit
    
    # Align HTF Camarilla levels to LTF
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Volume MA needs 20, EMA needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r1_aligned[i-1]  # Break above prior period R1
        breakout_down = close[i] < s1_aligned[i-1]  # Break below prior period S1
        
        # Trend filter: price vs 12h EMA50
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + above 12h EMA50 + volume spike
            if breakout_up and above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down + below 12h EMA50 + volume spike
            elif breakout_down and below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price retests the Camarilla pivot level (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls back to or below pivot level
                if close[i] <= pivot_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price rises back to or above pivot level
                if close[i] >= pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0