#!/usr/bin/env python3
"""
12h_1w_1d_camarilla_breakout_v1
- Timeframe: 12h
- Hypothesis: Camarilla pivot levels on 1d provide strong support/resistance.
  Breakout of these levels with volume confirmation and 1w trend filter
  works in both bull and bear markets by filtering with higher timeframe trend.
  Target: 20-50 trades per year (~80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day (to avoid look-ahead)
    # H = high, L = low, C = close
    H = high_1d
    L = low_1d
    C = close_1d
    
    # Camarilla levels
    # Resistance levels
    R4 = C + ((H - L) * 1.5000)
    R3 = C + ((H - L) * 1.2500)
    R2 = C + ((H - L) * 1.1666)
    R1 = C + ((H - L) * 1.0833)
    # Support levels
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    
    # Pivot point
    PP = (H + L + C) / 3
    
    # Arrays for each level
    r4 = R4
    r3 = R3
    r2 = R2
    r1 = R1
    pp = PP
    s1 = S1
    s2 = S2
    s3 = S3
    s4 = S4
    
    # Align to 12h timeframe (use previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 60 to ensure sufficient data
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions using Camarilla levels (using previous bar's aligned values)
        # Long: break above R3 with volume and uptrend
        long_breakout = close[i] > r3_aligned[i-1]
        # Short: break below S3 with volume and downtrend
        short_breakout = close[i] < s3_aligned[i-1]
        
        # Long: breakout above R3 + volume + uptrend
        long_signal = long_breakout and vol_confirm and price_above_ema
        # Short: breakout below S3 + volume + downtrend
        short_signal = short_breakout and vol_confirm and price_below_ema
        
        # Exit conditions
        # Exit long if price breaks below S3 or trend changes or volume fails
        long_exit = (close[i] < s3_aligned[i-1]) or (not vol_confirm) or (close[i] < ema_50_1w_aligned[i])
        # Exit short if price breaks above R3 or trend changes or volume fails
        short_exit = (close[i] > r3_aligned[i-1]) or (not vol_confirm) or (close[i] > ema_50_1w_aligned[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals