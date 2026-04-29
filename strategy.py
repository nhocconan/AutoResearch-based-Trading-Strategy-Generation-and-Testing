#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above R1 AND close > 4h EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below S1 AND close < 4h EMA50 AND volume > 1.8x 20-bar avg
# Exit when price crosses 4h EMA50 (trend change)
# Uses 1h timeframe for precise entry timing with 4h for signal direction
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Discrete position sizing (0.20) to minimize fee churn
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to avoid overtrading

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_range = high_4h - low_4h
    r1 = close_4h + 1.1 * camarilla_range / 12
    s1 = close_4h - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: >1.8x 20-bar average volume (balanced to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session or any required data is NaN
        if not in_session[i] or (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
                                 np.isnan(s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 4h EMA50 (trend change)
            if curr_close < curr_ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 4h EMA50 (trend change)
            if curr_close > curr_ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above R1 AND close > 4h EMA50 AND volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema50_4h and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S1 AND close < 4h EMA50 AND volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema50_4h and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals