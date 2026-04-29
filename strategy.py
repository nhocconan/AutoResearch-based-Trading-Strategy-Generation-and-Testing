#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above R1 AND close > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below S1 AND close < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit when price crosses 12h EMA50 (trend change)
# Uses discrete position sizing (0.30) to balance capture and risk.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to avoid overtrading.
# Camarilla R1/S1 levels provide tight support/resistance for frequent but filtered entries.
# Volume spike confirms participation, reducing false breakouts.
# 12h EMA50 trend filter ensures alignment with medium-term direction, working in both bull and bear regimes.

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter and Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 12h bar
    # Camarilla formula: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_range = high_12h - low_12h
    r1 = close_12h + 1.1 * camarilla_range / 12
    s1 = close_12h - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe (use previous 12h bar's levels)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume confirmation: >1.8x 20-bar average volume (balanced to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 12h EMA50 (trend change)
            if curr_close < curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above 12h EMA50 (trend change)
            if curr_close > curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above R1 AND close > 12h EMA50 AND volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema50_12h and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S1 AND close < 12h EMA50 AND volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema50_12h and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals