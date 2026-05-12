#!/usr/bin/env python3
# 4h_HTF_Swing_Structure_Volume
# Hypothesis: Combines daily swing high/low structure with 1-week trend filter and volume confirmation.
# Long when price breaks above prior daily swing high with volume spike and weekly uptrend.
# Short when price breaks below prior daily swing low with volume spike and weekly downtrend.
# Exit on opposite swing break. Designed for 20-35 trades/year to avoid fee drag while capturing
# major trend moves in both bull and bear markets via structure breaks.

name = "4h_HTF_Swing_Structure_Volume"
timeframe = "4h"
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
    
    # Load daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily swing points (using 3-bar lookback/forward for pivot points)
    # Swing high: high[i] > high[i-1] and high[i] > high[i+1]
    # Swing low: low[i] < low[i-1] and low[i] < low[i+1]
    swing_high = np.zeros(len(high_1d), dtype=bool)
    swing_low = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(1, len(high_1d)-1):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1]:
            swing_high[i] = True
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1]:
            swing_low[i] = True
    
    # Get most recent swing high and low values
    last_swing_high = np.full(len(high_1d), np.nan)
    last_swing_low = np.full(len(low_1d), np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    for i in range(len(high_1d)):
        if swing_high[i]:
            last_high_val = high_1d[i]
        if swing_low[i]:
            last_low_val = low_1d[i]
        last_swing_high[i] = last_high_val
        last_swing_low[i] = last_low_val
    
    # Align swing levels to 4h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, last_swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, last_swing_low)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        swing_high_val = swing_high_aligned[i]
        swing_low_val = swing_low_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        vol_confirm = volume_confirm[i]
        close_price = close[i]
        
        if position == 0:
            # LONG: Price breaks above daily swing high with volume and weekly uptrend
            if not np.isnan(swing_high_val) and close_price > swing_high_val and vol_confirm and close_price > ema50_val:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below daily swing low with volume and weekly downtrend
            elif not np.isnan(swing_low_val) and close_price < swing_low_val and vol_confirm and close_price < ema50_val:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below daily swing low (contrary signal)
            if not np.isnan(swing_low_val) and close_price < swing_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above daily swing high (contrary signal)
            if not np.isnan(swing_high_val) and close_price > swing_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals