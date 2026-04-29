#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below S3 AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit when price crosses opposite Camarilla level (R3/S3) or reverses across EMA50
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h to minimize fee drag.
# Camarilla provides structure; 12h EMA50 filters counter-trend moves in bear markets.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in both bull (trend continuation) and bear (mean reversion within trend) regimes.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 12h bar (standard formula)
    # Camarilla: based on prior period's high, low, close
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Calculate levels using prior 12h bar (to avoid look-ahead)
    # Shift by 1 to use only completed prior 12h bar
    h_12h_shifted = np.roll(h_12h, 1)
    l_12h_shifted = np.roll(l_12h, 1)
    c_12h_shifted = np.roll(c_12h, 1)
    h_12h_shifted[0] = np.nan
    l_12h_shifted[0] = np.nan
    c_12h_shifted[0] = np.nan
    
    # Camarilla calculations
    camarilla_range = (h_12h_shifted - l_12h_shifted)
    r3 = c_12h_shifted + (camarilla_range * 1.1 / 4)
    s3 = c_12h_shifted - (camarilla_range * 1.1 / 4)
    r4 = c_12h_shifted + (camarilla_range * 1.1 / 2)
    s4 = c_12h_shifted - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Get 12h data for EMA50 trend filter
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for Camarilla shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_12h_aligned[i]
        
        # Camarilla levels
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion) OR closes below EMA50 (trend reversal)
            if curr_close < s3_level or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion) OR closes above EMA50 (trend reversal)
            if curr_close > r3_level or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 12h EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 12h EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals