#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide institutional support/resistance. Breakouts above R1 or below S1
# with 4h trend alignment (price > EMA50 for longs, < EMA50 for shorts) capture institutional flow.
# Volume spike (>1.8x 20 EMA) confirms breakout legitimacy. Session filter (08-20 UTC) reduces noise.
# Discrete sizing 0.20 limits risk. Target: 80-120 total trades over 4 years (20-30/year).
# Works in bull/bear: uses 4h trend filter to align with higher timeframe direction.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    close_4h = pd.Series(df_4h['close'])
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (completed 4h bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h Camarilla pivot points (based on previous day's OHLC)
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1, H5, L5
    # We focus on R1 (H1) and S1 (L1) for breakouts
    # H1 = Close + 1.1*(High-Low)/12
    # L1 = Close - 1.1*(High-Low)/12
    # Using daily OHLC for Camarilla calculation
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session and volume confirmation
        volume_confirm = volume[i] > (1.8 * vol_ema_20[i])
        session_ok = in_session[i]
        
        if position == 0:
            # Long conditions: break above R1 + uptrend + volume spike + session
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema50_aligned[i] and 
                volume_confirm and session_ok):
                signals[i] = 0.20
                position = 1
            # Short conditions: break below S1 + downtrend + volume spike + session
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema50_aligned[i] and 
                  volume_confirm and session_ok):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to EMA50 OR volume drops OR session ends
            if (close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i] or 
                not in_session[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to EMA50 OR volume drops OR session ends
            if (close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i] or 
                not in_session[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals