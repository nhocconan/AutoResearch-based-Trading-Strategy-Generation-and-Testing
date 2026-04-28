# 4h Camarilla R1/S1 Breakout with 12h EMA50 Trend Filter and Volume Spike Confirmation
# Camarilla levels (R1/S1) act as strong intraday support/resistance; breaks indicate institutional participation.
# EMA50 on 12h filters for higher timeframe trend alignment, avoiding counter-trend trades.
# Volume spike (2x 24-period average) confirms breakout validity, reducing false signals.
# Works in bull markets (catching uptrends via R1 breakouts) and bear markets (catching downtrends via S1 breakdowns).
# Targets 75-200 total trades over 4 years (19-50/year) with discrete position sizing to minimize fee drag.

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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use R1/S1 as entry triggers (tighter levels for fewer trades)
    cam_high = df_1d['high'].values
    cam_low = df_1d['low'].values
    cam_close = df_1d['close'].values
    
    camarilla_width = (cam_high - cam_low) * 1.1
    r1 = cam_close + camarilla_width / 6  # R1 = C + (H-L)*1.1/6
    s1 = cam_close - camarilla_width / 6  # S1 = C - (H-L)*1.1/6
    
    # Align Camarilla levels to 4h timeframe (wait for prior day's close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 2x 24-period average (4 days of 4h bars)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA(50)
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Camarilla breakout conditions
        breakout_r1 = high[i] > r1_aligned[i-1]  # Break above R1
        breakdown_s1 = low[i] < s1_aligned[i-1]  # Break below S1
        
        # Entry conditions with volume spike confirmation
        long_entry = uptrend and breakout_r1 and volume_spike[i]
        short_entry = downtrend and breakdown_s1 and volume_spike[i]
        
        # Exit conditions: trend reversal or opposite Camarilla break
        long_exit = (not uptrend) or breakdown_s1
        short_exit = (not downtrend) or breakout_r1
        
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

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0