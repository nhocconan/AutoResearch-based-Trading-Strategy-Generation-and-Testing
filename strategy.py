#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_Reversion_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (S1, R1, S3, R3) from daily OHLC
    def calculate_camarilla(high, low, close):
        """Calculate Camarilla pivot levels: S1, R1, S3, R3"""
        range_hl = high - low
        # Camarilla formulas
        S3 = close - (range_hl * 1.1 / 2)
        S1 = close - (range_hl * 1.1 / 4)
        R1 = close + (range_hl * 1.1 / 4)
        R3 = close + (range_hl * 1.1 / 2)
        return S1, R1, S3, R3
    
    # Calculate Camarilla levels for each day
    camarilla_S1 = np.full(len(close_1d), np.nan)
    camarilla_R1 = np.full(len(close_1d), np.nan)
    camarilla_S3 = np.full(len(close_1d), np.nan)
    camarilla_R3 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        S1, R1, S3, R3 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_S1[i] = S1
        camarilla_R1[i] = R1
        camarilla_S3[i] = S3
        camarilla_R3[i] = R3
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1, additional_delay_bars=1)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1, additional_delay_bars=1)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3, additional_delay_bars=1)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3, additional_delay_bars=1)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches S3 and bounces, in daily uptrend, volume spike
            long_cond = (close[i] <= camarilla_S3_aligned[i] * 1.001 and  # Allow small buffer
                        close[i] > camarilla_S3_aligned[i] and
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price touches R3 and reverses, in daily downtrend, volume spike
            short_cond = (close[i] >= camarilla_R3_aligned[i] * 0.999 and  # Allow small buffer
                         close[i] < camarilla_R3_aligned[i] and
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches S1 or trend changes
            if (close[i] >= camarilla_S1_aligned[i] or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R1 or trend changes
            if (close[i] <= camarilla_R1_aligned[i] or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals