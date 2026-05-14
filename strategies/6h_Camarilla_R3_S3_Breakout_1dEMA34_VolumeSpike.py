#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Camarilla levels (R3/S3) act as strong intraday support/resistance; breaks indicate institutional participation.
# EMA34 on 1d filters for higher timeframe trend alignment, avoiding counter-trend trades.
# Volume spike (2x 24-period average) confirms breakout validity, reducing false signals.
# Works in bull markets (catching uptrends via R3 breakouts) and bear markets (catching downtrends via S3 breakdowns).
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use R3/S3 as entry triggers
    cam_high = df_1d['high'].values
    cam_low = df_1d['low'].values
    cam_close = df_1d['close'].values
    
    camarilla_width = (cam_high - cam_low) * 1.1
    r3 = cam_close + camarilla_width / 4
    s3 = cam_close - camarilla_width / 4
    
    # Align Camarilla levels to 6h timeframe (wait for prior day's close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: volume > 2x 24-period average (4 days of 6h bars)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_r3 = high[i] > r3_aligned[i-1]  # Break above R3
        breakdown_s3 = low[i] < s3_aligned[i-1]  # Break below S3
        
        # Entry conditions with volume spike confirmation
        long_entry = uptrend and breakout_r3 and volume_spike[i]
        short_entry = downtrend and breakdown_s3 and volume_spike[i]
        
        # Exit conditions: trend reversal or opposite Camarilla break
        long_exit = (not uptrend) or breakdown_s3
        short_exit = (not downtrend) or breakout_r3
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
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
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0