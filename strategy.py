#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_R1S1_Breakout_Volume_Control"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d: Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    
    # Calculate pivots using previous day's data
    P = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    S1 = P - (1.1/12) * (high_1d_prev - low_1d_prev)
    S2 = P - (1.1/6) * (high_1d_prev - low_1d_prev)
    S3 = P - (1.1/4) * (high_1d_prev - low_1d_prev)
    R1 = P + (1.1/12) * (high_1d_prev - low_1d_prev)
    R2 = P + (1.1/6) * (high_1d_prev - low_1d_prev)
    R3 = P + (1.1/4) * (high_1d_prev - low_1d_prev)
    
    # Align to 12h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    
    # === 1w: Trend filter (EMA50) ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema50_1w_aligned[i]
        s1_val = S1_aligned[i]
        s2_val = S2_aligned[i]
        r1_val = R1_aligned[i]
        r2_val = R2_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(s1_val) or np.isnan(s2_val) or \
           np.isnan(r1_val) or np.isnan(r2_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S1 with volume confirmation in uptrend
            if (close_val > s1_val and          # Break above S1 support
                close_val > ema_val and         # Above 1w EMA50 (uptrend)
                vol_ratio_val > 1.5):           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with volume confirmation in downtrend
            elif (close_val < r1_val and        # Break below R1 resistance
                  close_val < ema_val and       # Below 1w EMA50 (downtrend)
                  vol_ratio_val > 1.5):         # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S2 or trend reversal
            if (close_val < s2_val or           # Break below S2 support
                close_val < ema_val):           # Below 1w EMA50 (trend reversal)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R2 or trend reversal
            if (close_val > r2_val or           # Break above R2 resistance
                close_val > ema_val):           # Above 1w EMA50 (trend reversal)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals