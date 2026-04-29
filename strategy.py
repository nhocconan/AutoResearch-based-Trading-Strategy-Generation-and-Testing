#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume spike confirmation
# Long when price breaks above R1 AND close > 1w EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below S1 AND close < 1w EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses 1w EMA34 (trend change)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years) to avoid overtrading.
# Camarilla R1/S1 levels provide tight support/resistance for filtered entries.
# Volume spike confirms participation, reducing false breakouts.
# 1w EMA34 trend filter ensures alignment with long-term trend, working in both bull and bear regimes.

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter and Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 1w bar
    # Camarilla formula: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_range = high_1w - low_1w
    r1 = close_1w + 1.1 * camarilla_range / 12
    s1 = close_1w - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1d timeframe (use previous 1w bar's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: >2.0x 20-bar average volume (tight to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 1w EMA34 (trend change)
            if curr_close < curr_ema34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1w EMA34 (trend change)
            if curr_close > curr_ema34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R1 AND close > 1w EMA34 AND volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema34_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 AND close < 1w EMA34 AND volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema34_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals