#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 4h timeframe, use Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20-period average). Go long when price breaks above R1 with bullish 1d trend (close > 1d EMA34) and volume spike. Go short when price breaks below S1 with bearish 1d trend (close < 1d EMA34) and volume spike. Exit when price re-enters (R1,S1) range. Designed for 19-50 trades/year on 4h by requiring multi-timeframe alignment and volume confirmation, reducing fee drag while capturing strong trending moves in both bull and bear markets.
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_R1 = df_1d['close'].values + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 12
    camarilla_S1 = df_1d['close'].values - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 12
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed as they're based on completed 1d bar)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 warmup + volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike
            long_signal = (close[i] > camarilla_R1_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            
            # Short: price breaks below S1 + 1d downtrend + volume spike
            short_signal = (close[i] < camarilla_S1_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
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
            # Exit: price re-enters (R1,S1) range OR 1d trend turns down
            if (close[i] < camarilla_R1_aligned[i] and close[i] > camarilla_S1_aligned[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters (R1,S1) range OR 1d trend turns up
            if (close[i] < camarilla_R1_aligned[i] and close[i] > camarilla_S1_aligned[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0