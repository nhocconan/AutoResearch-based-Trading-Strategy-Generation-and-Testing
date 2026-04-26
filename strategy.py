#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v1
Hypothesis: Trade 12h breakouts from Camarilla R1/S1 levels with 1d EMA34 trend filter and volume confirmation. Works in bull/bear via trend filter; Camarilla levels provide structure in ranging markets. Discrete size 0.30 limits fee drag. Target 12-37 trades/year. Uses proper 2-bar breakout confirmation and exits on trend reversal.
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
    
    # Get 1d data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla pivot levels (R1, S1) from previous 1d bar
    # Formula: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    # Using previous bar's OHLC to avoid look-ahead
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # first bar has no previous
    
    # Calculate Camarilla R1 and S1 for previous 1d bar
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 12
    
    # 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d data to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average on 12h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Require close to stay beyond level for 2 consecutive bars to reduce false breakouts
    close_above_r1 = close > camarilla_r1_aligned
    close_below_s1 = close < camarilla_s1_aligned
    close_above_r1_2bar = close_above_r1 & np.roll(close_above_r1, 1)
    close_below_s1_2bar = close_below_s1 & np.roll(close_below_s1, 1)
    close_above_r1_2bar[0] = False
    close_below_s1_2bar[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + 1d uptrend + 2-bar confirmation
            long_breakout = close_above_r1_2bar[i]
            long_signal = long_breakout and volume_spike[i] and trend_1d_uptrend
            
            # Short: price breaks below S1 + volume spike + 1d downtrend + 2-bar confirmation
            short_breakout = close_below_s1_2bar[i]
            short_signal = short_breakout and volume_spike[i] and trend_1d_downtrend
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price touches S1 level OR 1d trend turns down
            if (close[i] < camarilla_s1_aligned[i] or not trend_1d_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price touches R1 level OR 1d trend turns up
            if (close[i] > camarilla_r1_aligned[i] or not trend_1d_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0