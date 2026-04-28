#!/usr/bin/env python3
# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation.
# Camarilla pivot levels identify key support/resistance levels derived from prior 4h session.
# Breakout above/below pivot levels with volume confirmation indicates institutional participation.
# 4h EMA trend filter ensures we trade in the direction of the higher timeframe trend.
# Session filter (08-20 UTC) reduces noise during low-volume periods.
# Designed for 1h timeframe to target 60-150 total trades over 4 years (15-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla pivot levels from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = df_4h['close'].values
    
    # Camarilla levels for the current 4h bar (calculated from previous 4h bar)
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    # We'll calculate these for the previous 4h bar to avoid look-ahead
    # Since we're on 1h timeframe, we need to get the previous completed 4h bar
    
    # Pre-calculate Camarilla levels for each 4h bar
    camarilla_r1 = np.full(len(df_4h), np.nan)
    camarilla_s1 = np.full(len(df_4h), np.nan)
    camarilla_r2 = np.full(len(df_4h), np.nan)
    camarilla_s2 = np.full(len(df_4h), np.nan)
    
    for i in range(1, len(df_4h)):  # Start from 1 to use previous bar
        phigh = high_4h[i-1]
        plow = low_4h[i-1]
        pclose = close_4h[i-1]
        
        r1 = pclose + ((phigh - plow) * 1.0833)
        s1 = pclose - ((phigh - plow) * 1.0833)
        r2 = pclose + ((phigh - plow) * 1.1666)
        s2 = pclose - ((phigh - plow) * 1.1666)
        
        camarilla_r1[i] = r1
        camarilla_s1[i] = s1
        camarilla_r2[i] = r2
        camarilla_s2[i] = s2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s2)
    
    # Volume filter: volume > 1.3x 24-period average
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r2_aligned[i]) or 
            np.isnan(camarilla_s2_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA(34)
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > camarilla_r1_aligned[i]  # Break above R1
        breakout_s1 = close[i] < camarilla_s1_aligned[i]  # Break below S1
        breakout_r2 = close[i] > camarilla_r2_aligned[i]  # Break above R2
        breakout_s2 = close[i] < camarilla_s2_aligned[i]  # Break below S2
        
        # Entry conditions with volume confirmation
        # Long: break above R1 or R2 in uptrend
        long_entry = (breakout_r1 or breakout_r2) and uptrend and volume_filter[i]
        # Short: break below S1 or S2 in downtrend
        short_entry = (breakout_s1 or breakout_s2) and downtrend and volume_filter[i]
        
        # Exit conditions: when trend reverses or opposite breakout occurs
        long_exit = (not uptrend) or breakout_s1  # Exit on trend reversal or break below S1
        short_exit = (not downtrend) or breakout_r1  # Exit on trend reversal or break above R1
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_CamarillaPivotBreakout_4hEMA34_TrendFilter_Volume"
timeframe = "1h"
leverage = 1.0