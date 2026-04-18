#!/usr/bin/env python3
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
    
    # Get daily data for pivot points and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot Point and S1, R1 (standard formula)
    P = np.full_like(high_1d, np.nan)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        P[i] = (prev_high + prev_low + prev_close) / 3.0
        R1[i] = 2 * P[i] - prev_low
        S1[i] = 2 * P[i] - prev_high
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all data to 12h timeframe
    P_12h = align_htf_to_ltf(prices, df_1d, P)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volatility filter: ATR(14) < 0.5 * ATR(50) indicates low volatility (range)
    def calculate_atr(high, low, close, period):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr50_1d = calculate_atr(high_1d, low_1d, close_1d, 50)
    atr14_12h = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_12h = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(P_12h[i]) or np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(ema_1w_12h[i]) or np.isnan(atr14_12h[i]) or np.isnan(atr50_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: low volatility regime (range)
        low_vol = atr14_12h[i] < 0.5 * atr50_12h[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price touches S1 with volume in low volatility
            if close[i] <= S1_12h[i] and vol_confirm and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 with volume in low volatility
            elif close[i] >= R1_12h[i] and vol_confirm and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches P or volatility increases
            if close[i] >= P_12h[i] or not low_vol:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches P or volatility increases
            if close[i] <= P_12h[i] or not low_vol:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_Touch_Volume_LowVol"
timeframe = "12h"
leverage = 1.0