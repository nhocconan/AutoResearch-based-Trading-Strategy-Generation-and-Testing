#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d EMA34 trend filter and volume confirmation (1.5x 20-bar avg). In trending markets (price > EMA34), breakouts in direction of trend capture momentum. Volume confirms breakout validity. Designed for 12h timeframe targeting 12-37 trades/year. Works in bull/bear by only taking trend-aligned breakouts.
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
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe (1-day lagged for completed bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 1d data (based on previous day's OHLC)
    # Camarilla: R1 = C + ((H-L) * 1.1/12), S1 = C - ((H-L) * 1.1/12)
    camarilla_r1_1d = close_1d + ((df_1d['high'].values - df_1d['low'].values) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((df_1d['high'].values - df_1d['low'].values) * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 1d EMA34 trend with volume confirmation
            uptrend = close[i] > ema_34_aligned[i]
            downtrend = close[i] < ema_34_aligned[i]
            
            long_signal = (close[i] > camarilla_r1_aligned[i]) and uptrend and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla S1 (mean reversion to midpoint)
            exit_signal = close[i] < camarilla_s1_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla R1 (mean reversion to midpoint)
            exit_signal = close[i] > camarilla_r1_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0