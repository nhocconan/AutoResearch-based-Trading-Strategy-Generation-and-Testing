#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Uses 4h Camarilla levels (R1/S1) for direction, 4h EMA20 for trend filter, and volume spike confirmation on 1h chart. 
# Entry only during 08-20 UTC to avoid low-liquidity hours. Targets 15-30 trades/year by requiring confluence of trend, level break, and volume.
# Works in bull/bear markets via 4h trend filter and volume confirmation to avoid false breakouts.

timeframe = "1h"
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
leverage = 1.0

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
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 4h Camarilla levels
    c_high = df_4h['high'].values
    c_low = df_4h['low'].values
    c_close = df_4h['close'].values
    
    camarilla_r1 = c_close + 1.1 * (c_high - c_low) / 12
    camarilla_s1 = c_close - 1.1 * (c_high - c_low) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume spike: 2x 24-period MA (approx 1 day on 1h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 20)  # Ensure volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or not in session
        if (not in_session[i] or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > R1 with volume spike and 4h uptrend
            if close[i] > camarilla_r1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: close < S1 with volume spike and 4h downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_20_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: touch S1 (opposite level) or trend failure
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: touch R1 (opposite level) or trend failure
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals