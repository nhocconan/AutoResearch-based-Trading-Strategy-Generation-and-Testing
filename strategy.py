#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Camarilla pivot breakout on 4h with 12h EMA50 trend filter and volume confirmation. Long when price breaks above R1 with 12h EMA50 uptrend and volume > 1.5x average, short when price breaks below S1 with 12h EMA50 downtrend and volume confirmation. Uses discrete sizing (0.30) to minimize fees. Designed for 20-40 trades/year, works in bull markets via breakout longs and in bear markets via breakdown shorts. Camarilla levels provide institutional support/resistance, EMA50 filters counter-trend noise, volume confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA50 on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    camarilla_range = (high_1d - low_1d)
    r1_1d = close_1d + 1.1 * camarilla_range / 12
    s1_1d = close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed as they're based on completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 12h EMA50 uptrend + volume spike
            long_signal = (close[i] > r1_aligned[i]) and (close[i-1] <= r1_aligned[i-1]) and \
                         (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and volume_spike[i]
            # Short: price breaks below S1 + 12h EMA50 downtrend + volume spike
            short_signal = (close[i] < s1_aligned[i]) and (close[i-1] >= s1_aligned[i-1]) and \
                          (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit when price closes below S1 (breakdown below support)
            exit_signal = close[i] < s1_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit when price closes above R1 (breakout above resistance)
            exit_signal = close[i] > r1_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0