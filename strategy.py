#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_AntiTrend_v1
4h strategy using Camarilla pivot R1/S1 levels with volume confirmation and counter-trend bias.
- Long: Close below S1 (support) + volume > 1.5x 20-period mean + price > 200 EMA (mean reversion in uptrend)
- Short: Close above R1 (resistance) + volume > 1.5x 20-period mean + price < 200 EMA (mean reversion in downtrend)
- Exit: Opposite condition or trend reversal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using standard Camarilla formulas:
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    # We'll use R1 and S1 as primary levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot components
    hl_range = high_1d - low_1d
    r1 = close_1d + (hl_range * 1.0833)
    s1 = close_1d - (hl_range * 1.0833)
    
    # Align daily R1/S1 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 4h EMA200 for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition
        uptrend = close[i] > ema_200[i]
        downtrend = close[i] < ema_200[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Mean reversion conditions at Camarilla levels
        touch_s1 = low[i] <= s1_aligned[i]  # price touches or goes below S1
        touch_r1 = high[i] >= r1_aligned[i]  # price touches or goes above R1
        
        if position == 0:
            # Long: price at S1 support + volume + uptrend (buy dip in uptrend)
            if touch_s1 and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price at R1 resistance + volume + downtrend (sell rally in downtrend)
            elif touch_r1 and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price at R1 resistance or trend reversal
            if touch_r1 or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price at S1 support or trend reversal
            if touch_s1 or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_AntiTrend_v1"
timeframe = "4h"
leverage = 1.0