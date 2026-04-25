#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA50 trend filter and volume spike (2.0x 20-bar avg). 
Trade breakouts aligned with 1d EMA50 trend to capture momentum while avoiding counter-trend whipsaws. 
Volume confirms institutional participation. Designed for 4h timeframe targeting 20-50 trades/year.
Works in bull/bear by following 1d EMA50 trend - long when price > EMA50, short when price < EMA50.
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
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe (1-day lagged for completed bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 1d data (based on previous day's OHLC)
    # Camarilla: R1 = C + ((H-L) * 1.1/12), S1 = C - ((H-L) * 1.1/12)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume (80 periods = ~6.7h on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and Camarilla
    start_idx = max(30, 50)  # 50 for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend: price > EMA50 = bullish, price < EMA50 = bearish
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Look for breakout signals at R1/S1 with volume confirmation and trend alignment
            long_signal = (close[i] > camarilla_r1_aligned[i]) and volume_spike[i] and trend_bullish
            short_signal = (close[i] < camarilla_s1_aligned[i]) and volume_spike[i] and trend_bearish
            
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
            # Exit when price breaks below Camarilla S1 or trend reverses (price < EMA50)
            exit_signal = (close[i] < camarilla_s1_aligned[i]) or (close[i] < ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above Camarilla R1 or trend reverses (price > EMA50)
            exit_signal = (close[i] > camarilla_r1_aligned[i]) or (close[i] > ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0