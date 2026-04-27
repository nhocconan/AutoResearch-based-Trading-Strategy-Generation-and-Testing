#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels (R1, S1) from 12h timeframe for breakout entries, filtered by 12h EMA50 trend and volume spike (>2x average). Enters long when price breaks above 12h R1 AND 12h close > 12h EMA50 (uptrend) AND volume > 2x average. Enters short when price breaks below 12h S1 AND 12h close < 12h EMA50 (downtrend) AND volume > 2x average. Exits when price reverts to 12h close (mean reversion) OR trend breaks. Designed for 4h timeframe with moderate entries to avoid fee drag: target 20-50 trades/year. Camarilla levels provide high-probability reversal/breakout points, and volume confirmation avoids low-conviction moves. Works in both bull and bear markets via 12h trend filter and volume confirmation to avoid false signals.
"""

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
    
    # Get 12h data for Camarilla pivot calculation and EMA50
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels for 12h timeframe
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    hl_range_12h = df_12h['high'].values - df_12h['low'].values
    camarilla_r1_12h = df_12h['close'].values + hl_range_12h * 1.1 / 12
    camarilla_s1_12h = df_12h['close'].values - hl_range_12h * 1.1 / 12
    
    # Align 12h indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    camarilla_r1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    
    # Volume confirmation: current volume > 2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 12h EMA50 (50), volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_12h_aligned[i]) or np.isnan(camarilla_s1_12h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_12h_val = ema_50_12h_aligned[i]
        r1_12h_val = camarilla_r1_12h_aligned[i]
        s1_12h_val = camarilla_s1_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: price breakout above R1 (long) or below S1 (short) with trend and volume
            # Long: price > R1 AND 12h uptrend AND volume confirmation
            long_condition = (close_val > r1_12h_val) and (close_val > ema_12h_val) and vol_conf
            # Short: price < S1 AND 12h downtrend AND volume confirmation
            short_condition = (close_val < s1_12h_val) and (close_val < ema_12h_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to 12h close (mean reversion) OR trend breaks
            close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['close'].values)
            exit_condition = (close_val <= close_12h_aligned[i]) or (close_val < ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to 12h close (mean reversion) OR trend breaks
            close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['close'].values)
            exit_condition = (close_val >= close_12h_aligned[i]) or (close_val > ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0