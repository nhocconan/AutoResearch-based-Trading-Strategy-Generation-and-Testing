#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 level AND 1d trend is up (close > EMA34) AND volume > 1.5x 20-period average volume; enter short when price breaks below Camarilla S1 level AND 1d trend is down (close < EMA34) AND volume > 1.5x 20-period average volume. Uses Camarilla pivot levels from 1d as key support/resistance, 1d EMA34 for trend filter, and volume confirmation to avoid false breakouts. Targets 12-37 trades per year over 4 years with discrete sizing (0.0, ±0.25) to minimize fee churn. Works in bull via trend continuation breakouts and in bear via counter-trend reversals at pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    if len(df_12h) < 2:  # need at least 1 bar for pivot calculation
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 12h data (using previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # Camarilla levels: based on previous bar's range
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    range_12h = high_12h - low_12h
    camarilla_r1 = close_12h + range_12h * 1.1 / 12
    camarilla_s1 = close_12h - range_12h * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (already on 12h, but align for consistency)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla warmup (1 bar), EMA warmup (34 bars), volume SMA warmup (20 bars)
    start_idx = max(1, 34, 20)  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_sma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + 1d uptrend + volume confirmation
            long_signal = close[i] > camarilla_r1_aligned[i] and trend_uptrend and volume_confirm
            
            # Short: price breaks below Camarilla S1 + 1d downtrend + volume confirmation
            short_signal = close[i] < camarilla_s1_aligned[i] and trend_downtrend and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Camarilla S1 OR trend change to downtrend
            if close[i] < camarilla_s1_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Camarilla R1 OR trend change to uptrend
            if close[i] > camarilla_r1_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0