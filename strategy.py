#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla pivot levels from daily timeframe provide strong intraday support/resistance.
Buy breakouts above R1 with volume confirmation in uptrend (price above 1d EMA34).
Sell breakdowns below S1 with volume confirmation in downtrend (price below 1d EMA34).
Uses 4h timeframe for execution with 1d trend filter and volume confirmation.
Target: 20-30 trades per year (~80-120 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    r1 = prev_close + (range_hl * 1.1 / 12)
    s1 = prev_close - (range_hl * 1.1 / 12)
    
    # Align daily Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA34
        uptrend_regime = close[i] > ema_34_1d_aligned[i]
        downtrend_regime = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: price breaks above R1 in uptrend regime + volume
            long_entry = (close[i] > r1_aligned[i]) and uptrend_regime and volume_confirm
            # Short: price breaks below S1 in downtrend regime + volume
            short_entry = (close[i] < s1_aligned[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below pivot or regime changes to downtrend
            if (close[i] < pivot_aligned[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above pivot or regime changes to uptrend
            if (close[i] > pivot_aligned[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals