#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Reversal_v1
Hypothesis: Camarilla pivot levels (R1/S1) from 1d act as reversal zones in mean-reverting markets.
Combined with 1d trend filter (EMA34) and volume confirmation to avoid counter-trend trades.
Designed for low trade frequency (target 20-50/year) to minimize fee drag.
Works in both bull and bear markets by fading extremes in ranging conditions and following trend in strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_r2 = np.zeros_like(close_1d)
    camarilla_s2 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Use previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        rang = prev_high - prev_low
        
        camarilla_r1[i] = prev_close + (rang * 1.1 / 12)
        camarilla_s1[i] = prev_close - (rang * 1.1 / 12)
        camarilla_r2[i] = prev_close + (rang * 1.1 / 6)
        camarilla_s2[i] = prev_close - (rang * 1.1 / 6)
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align 1d indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough 1d data for indicators
    start_idx = 34  # EMA34 needs 34 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r2 = camarilla_r2_aligned[i]
        s2 = camarilla_s2_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs EMA34 (1d)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            # Long setup: price at S1/S2 support in uptrend or strong reversal from S2 in downtrend
            if (uptrend and close_val <= s1 and vol_conf) or \
               (not uptrend and close_val <= s2 and vol_conf):
                # Additional confirmation: price showing rejection (close > open)
                if close_val > prices['open'].iloc[i]:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            
            # Short setup: price at R1/R2 resistance in downtrend or strong reversal from R2 in uptrend
            elif (downtrend and close_val >= r1 and vol_conf) or \
                 (not downtrend and close_val >= r2 and vol_conf):
                # Additional confirmation: price showing rejection (close < open)
                if close_val < prices['open'].iloc[i]:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        
        elif position == 1:
            # Exit: price reaches R1 (profit target) or breaks below S2 (stop/reversal)
            if close_val >= r1 or close_val < s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        
        elif position == -1:
            # Exit: price reaches S1 (profit target) or breaks above R2 (stop/reversal)
            if close_val <= s1 or close_val > r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Reversal_v1"
timeframe = "4h"
leverage = 1.0