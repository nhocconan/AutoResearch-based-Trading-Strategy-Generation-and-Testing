#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_R1S1_Breakout_VolumeTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Camarilla pivot from previous 4h bar (use H, L, C of previous 4h bar)
    # Camarilla: P = (H + L + C) / 3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    camarilla_p = (h_4h + l_4h + c_4h) / 3
    camarilla_range = h_4h - l_4h
    camarilla_r1 = c_4h + camarilla_range * 1.1 / 12
    camarilla_s1 = c_4h - camarilla_range * 1.1 / 12
    
    # Align 4h Camarilla levels to 1h timeframe (use previous 4h bar's values)
    p_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_p)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 1d trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1h volume spike: 24-period average (24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(p_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_4h_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_4h_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below pivot or volume drops
            if close[i] < p_4h_aligned[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above pivot or volume drops
            if close[i] > p_4h_aligned[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h pivot reference and 1d trend filter
# - Uses 4h Camarilla levels (S1/R1) as dynamic support/resistance
# - Breakout above S1 with volume spike in daily uptrend = long
# - Breakdown below R1 with volume spike in daily downtrend = short
# - Volume confirmation (2x average) filters false breakouts
# - Session filter (8-20 UTC) reduces noise outside active hours
# - Position size 0.20 targets 15-37 trades/year on 1h timeframe
# - Works in bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to 4h pivot point or volume weakens
# - Designed to avoid overtrading while capturing meaningful moves