#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Camarilla levels (R4/S4) represent extreme intraday support/resistance; breaks indicate strong momentum.
# EMA50 on 1d filters for higher timeframe trend alignment, avoiding counter-trend trades.
# Volume spike (2x 24-period average of 4h bars = 4 days) confirms breakout validity.
# Works in bull markets (catching uptrends via R4 breakouts) and bear markets (catching downtrends via S4 breakdowns).
# Targets 30-80 total trades over 4 years (8-20/year) with discrete position sizing to minimize fee drag.

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
    
    # Get 1d data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use R4/S4 as entry triggers for stronger breakouts
    cam_high = df_1d['high'].values
    cam_low = df_1d['low'].values
    cam_close = df_1d['close'].values
    
    camarilla_width = (cam_high - cam_low) * 1.1
    r4 = cam_close + camarilla_width / 2
    s4 = cam_close - camarilla_width / 2
    
    # Align Camarilla levels to 4h timeframe (wait for prior day's close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: volume > 2x 24-period average (4 days of 4h bars)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_r4 = high[i] > r4_aligned[i-1]  # Break above R4
        breakdown_s4 = low[i] < s4_aligned[i-1]  # Break below S4
        
        # Entry conditions with volume spike confirmation
        long_entry = uptrend and breakout_r4 and volume_spike[i]
        short_entry = downtrend and breakdown_s4 and volume_spike[i]
        
        # Exit conditions: trend reversal or opposite Camarilla break
        long_exit = (not uptrend) or breakdown_s4
        short_exit = (not downtrend) or breakout_r4
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
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
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R4_S4_Breakout_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0