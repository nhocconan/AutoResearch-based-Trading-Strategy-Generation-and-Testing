#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses 12h Camarilla R3/S3 breakouts filtered by 1d EMA34 trend and volume spike (>2x average).
Enter long when price breaks above 12h R3 AND 1d close > 1d EMA34 (uptrend) AND volume > 2x average.
Enter short when price breaks below 12h S3 AND 1d close < 1d EMA34 (downtrend) AND volume > 2x average.
Exit when price retests the Camarilla pivot point (PP) or 1d trend reverses.
Designed for 12h timeframe with moderate entries to avoid fee drag: target 12-37 trades/year.
Works in both bull and bear markets via 1d trend filter and volume confirmation to avoid false signals.
Camarilla levels are calculated from prior 1d OHLC, providing institutional support/resistance levels.
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
    
    # Get 12h data for price action and Camarilla calculation (using prior 1d OHLC)
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for Camarilla calculation (prior day OHLC) and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d OHLC
    # Camarilla: based on previous day's range
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    # Prior 1d OHLC (already completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    rang = (high_1d - low_1d) * 1.1
    r3 = close_1d + (rang / 4)      # R3 level
    s3 = close_1d - (rang / 4)      # S3 level
    pp = (high_1d + low_1d + close_1d) / 3  # Pivot Point
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d bar available)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    pp_12h = align_htf_to_ltf(prices, df_1d, pp)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2 * 20-period average (using 12h volume)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20), Camarilla (need 1d data)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(pp_12h[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r3_val = r3_12h[i]
        s3_val = s3_12h[i]
        pp_val = pp_12h[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla breakout with trend and volume
            # Long: price > R3 AND 1d uptrend AND volume
            long_condition = (close_val > r3_val) and (close_val > ema_1d_val) and vol_conf
            # Short: price < S3 AND 1d downtrend AND volume
            short_condition = (close_val < s3_val) and (close_val < ema_1d_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price retests PP OR 1d trend breaks
            exit_condition = (close_val <= pp_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price retests PP OR 1d trend breaks
            exit_condition = (close_val >= pp_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0