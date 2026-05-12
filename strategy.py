#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Pullback_MeanReversion
# Hypothesis: In 4h timeframe, after price touches Camarilla R1/S1 levels, wait for pullback to the pivot point with volume confirmation for mean reversion entries.
# Long when price pulls back from R1 to pivot with volume > 1.5x average and RSI < 40.
# Short when price pulls back from S1 to pivot with volume > 1.5x average and RSI > 60.
# Exit on close beyond R1/S1 (breakout) or opposite pivot touch.
# Designed for low frequency (20-30 trades/year) to minimize fee drag and work in ranging markets.

name = "4h_Camarilla_R1_S1_Pullback_MeanReversion"
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
    
    # Load daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate pivot point and range
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla levels
    r1 = daily_pivot + daily_range * 1.083
    s1 = daily_pivot - daily_range * 1.083
    pivot = daily_pivot  # Pivot point
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # RSI(14) for overbought/oversold
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        rsi_val = rsi_values[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # LONG: Pullback from R1 to pivot with RSI < 40 and volume confirmation
            if (close[i] >= pivot_val * 0.995 and close[i] <= pivot_val * 1.005 and  # Near pivot
                low[i] < r1_val and  # Touched or went below R1 recently (check low)
                rsi_val < 40 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback from S1 to pivot with RSI > 60 and volume confirmation
            elif (close[i] >= pivot_val * 0.995 and close[i] <= pivot_val * 1.005 and  # Near pivot
                  high[i] > s1_val and  # Touched or went above S1 recently (check high)
                  rsi_val > 60 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks above R1 or closes below S1
            if close[i] > r1_val or close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks below S1 or closes above R1
            if close[i] < s1_val or close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals