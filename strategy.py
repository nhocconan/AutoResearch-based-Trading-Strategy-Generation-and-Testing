#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Use 4h trend (price > 4h EMA50) and 1d momentum (price > 1d EMA200) for direction. 
# Enter long at 1h when price breaks above Camarilla R1 with volume > 1.5x average, short below S1 in downtrend.
# Exit when price crosses Camarilla pivot (PP) or ATR-based stop hit. Designed for 15-35 trades/year on 1h.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Calculate True Range and ATR(20)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Calculate Camarilla levels from previous day
    # Using daily high, low, close from 1d data
    df_1d = get_htf_data(prices, '1d')
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Previous day values for Camarilla calculation
    pd_high = np.concatenate([[np.nan], d_high[:-1]])
    pd_low = np.concatenate([[np.nan], d_low[:-1]])
    pd_close = np.concatenate([[np.nan], d_close[:-1]])
    
    # Camarilla formulas
    pp = (pd_high + pd_low + pd_close) / 3
    r1 = pp + (pd_high - pd_low) * 1.1 / 12
    s1 = pp - (pd_high - pd_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA200 for regime filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume average (24 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.nanmean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend: 4h EMA50 and 1d EMA200 both must agree
            uptrend = close[i] > ema_50_4h_aligned[i] and close[i] > ema_200_1d_aligned[i]
            downtrend = close[i] < ema_50_4h_aligned[i] and close[i] < ema_200_1d_aligned[i]
            
            if uptrend:
                # Long: Break above R1 with volume confirmation
                if close[i] > r1_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.20
                    position = 1
            elif downtrend:
                # Short: Break below S1 with volume confirmation
                if close[i] < s1_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit: Price crosses below pivot PP or ATR stop
            if close[i] < pp_aligned[i] or (i > 0 and low[i] < pp_aligned[i] - 1.5 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price crosses above pivot PP or ATR stop
            if close[i] > pp_aligned[i] or (i > 0 and high[i] > pp_aligned[i] + 1.5 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals