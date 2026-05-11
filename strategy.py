#!/usr/bin/env python3
# 12h_1w_Camarilla_R1_S1_Breakout_TrendVolume
# Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 level from previous 1d with volume confirmation and 1w uptrend (price > 1w SMA50).
# Enter short when price breaks below Camarilla S1 level with volume confirmation and 1w downtrend (price < 1w SMA50).
# Exit when price crosses back through the Camarilla pivot point (central level) or trend reverses.
# Camarilla levels provide precise intraday support/resistance, 1w trend filters for major regime, volume avoids false breakouts.
# Target: 20-40 trades/year (80-160 total over 4 years) to stay within 12h limits and minimize fee drag.

name = "12h_1w_Camarilla_R1_S1_Breakout_TrendVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Calculate 1d Camarilla levels (based on previous day's OHLC) ---
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.1250)
    # R2 = close + ((high - low) * 0.7500)
    # R1 = close + ((high - low) * 0.5000)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 0.5000)
    # S2 = close - ((high - low) * 0.7500)
    # S3 = close - ((high - low) * 1.1250)
    # S4 = close - ((high - low) * 1.5000)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate levels for each 1d bar
    r1_1d = close_1d + ((high_1d - low_1d) * 0.5000)
    s1_1d = close_1d - ((high_1d - low_1d) * 0.5000)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align 1d levels to 12h timeframe (use previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # --- 1w trend filter (SMA50 on close) ---
    close_1w = df_1w['close'].values
    sma_1w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i-50:i])
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # --- Volume filter: volume > 1.5x 20-period average ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d levels (need at least 2 days for previous day), 1w SMA50, and volume MA
    start_idx = max(50, 20)  # 50 for 1w SMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(pp_1d_aligned[i]) or \
           np.isnan(sma_1w_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and 1w uptrend
            if close[i] > r1_1d_aligned[i] and vol_confirm and close[i] > sma_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and 1w downtrend
            elif close[i] < s1_1d_aligned[i] and vol_confirm and close[i] < sma_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below PP (reversion to mean) OR 1w trend turns down
                if close[i] < pp_1d_aligned[i] or close[i] < sma_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above PP (reversion to mean) OR 1w trend turns up
                if close[i] > pp_1d_aligned[i] or close[i] > sma_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals