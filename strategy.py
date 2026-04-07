#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray with 1d/1w trend filter and volume confirmation
# Hypothesis: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure.
# Combined with 1d/1w trend filters to avoid counter-trend trades, and volume to confirm strength.
# Works in bull via Bull Power > 0 with uptrend, in bear via Bear Power < 0 with downtrend.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.
name = "6h_elderay_1d1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA(13) for Elder Ray on 1d
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate EMA(50) for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA(50) for 1w trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume confirmation: 6h volume > 20-period EMA of volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters: 1d and 1w EMA50 alignment
        uptrend = (close[i] > ema50_1d_aligned[i]) and (close[i] > ema50_1w_aligned[i])
        downtrend = (close[i] < ema50_1d_aligned[i]) and (close[i] < ema50_1w_aligned[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ema20[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes positive (selling pressure gone) OR trend breaks
            if bear_power[i] > 0 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Bull Power becomes negative (buying pressure gone) OR trend breaks
            if bull_power[i] < 0 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Bull Power > 0 (buying pressure) + uptrend + volume confirmation
            if bull_power[i] > 0 and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power < 0 (selling pressure) + downtrend + volume confirmation
            elif bear_power[i] < 0 and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals