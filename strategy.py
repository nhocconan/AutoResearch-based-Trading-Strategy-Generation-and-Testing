#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA40 for trend filter
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Load daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from previous day (H4, L4, H3, L3)
    # Calculated as: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L), H3 = C + 1.125*(H-L), L3 = C - 1.125*(H-L)
    camarilla_H4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_L4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_H3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_L3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align Camarilla levels (need previous day's levels, so shift by 1)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4, additional_delay_bars=1)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4, additional_delay_bars=1)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3, additional_delay_bars=1)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3, additional_delay_bars=1)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_40_1w_aligned[i]) or 
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above H4 resistance + above weekly EMA40 + volume filter
            if high[i] > camarilla_H4_aligned[i] and close[i] > ema_40_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below L4 support + below weekly EMA40 + volume filter
            elif low[i] < camarilla_L4_aligned[i] and close[i] < ema_40_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below H3 or below weekly EMA40
            if low[i] < camarilla_H3_aligned[i] or close[i] < ema_40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above L3 or above weekly EMA40
            if high[i] > camarilla_L3_aligned[i] or close[i] > ema_40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals