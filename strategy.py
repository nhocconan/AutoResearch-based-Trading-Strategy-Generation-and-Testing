#!/usr/bin/env python3
"""
1h Camarilla R1/S1 Breakout + 4h EMA50 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) on 4h provide intraday support/resistance. 
Break above R1 with volume and 4h uptrend (EMA50) signals bullish momentum; 
break below S1 with volume and 4h downtrend signals bearish momentum. 
1h timeframe for precise entry timing, 4h for signal direction to reduce overtrading.
Session filter (08-20 UTC) avoids low-volume Asian session noise. Targets 15-37 trades/year.
Works in bull/bear via trend filter - only trades with 4h trend, avoiding chop.
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
    
    # Get 4h data for Camarilla pivots and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots on 4h (using previous 4h bar's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Shift by 1 to use previous completed 4h bar (no look-ahead)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    # First bar has no previous bar
    prev_close_4h[0] = np.nan
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    camarilla_r1_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12
    camarilla_s1_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12
    
    # Align to 1h timeframe (wait for 4h bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Start index: need enough for EMA50 warmup and proper alignment
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above R1 AND above 4h EMA50 (uptrend filter)
            long_condition = (curr_close > r1_level) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below S1 AND below 4h EMA50 (downtrend filter)
            short_condition = (curr_close < s1_level) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below S1 or trend breaks
            if curr_close < s1_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above R1 or trend breaks
            if curr_close > r1_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0