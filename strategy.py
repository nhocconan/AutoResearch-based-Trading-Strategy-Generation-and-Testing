#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Weekly_Pivot_R1S1_Breakout_VolumeATR_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly high, low, close from daily data (7 days per week)
    week_high = pd.Series(high).rolling(window=7, min_periods=7).max().values
    week_low = pd.Series(low).rolling(window=7, min_periods=7).min().values
    week_close = pd.Series(close).rolling(window=7, min_periods=7).last().values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot = (week_high + week_low + week_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    
    # Align weekly pivot levels to 12h timeframe (weekly bars align with 12h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Weekly ATR for volatility filter (14-period)
    tr = np.maximum(week_high[1:] - week_low[1:], np.absolute(week_high[1:] - week_close[:-1]))
    tr = np.maximum(tr, np.absolute(week_low[1:] - week_close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: current volume > 2.0x 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        piv = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr = atr_14_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < s1_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below pivot or ATR-based stop
            if price < piv or price < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above pivot or ATR-based stop
            if price > piv or price > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals