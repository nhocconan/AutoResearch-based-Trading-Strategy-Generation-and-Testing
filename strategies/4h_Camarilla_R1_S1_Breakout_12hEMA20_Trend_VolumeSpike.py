#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA20_Trend_VolumeSpike
Hypothesis: On 4h timeframe, use Camarilla R1/S1 levels from 1d for breakout entries, filtered by 12h trend direction (close > EMA20) and volume spike (>2.0x 20-period average). Enter long when price breaks above R1 with 12h uptrend and volume spike. Enter short when price breaks below S1 with 12h downtrend and volume spike. Uses discrete position size 0.25 to balance capture and drawdown. Designed for 19-50 trades/year on 4h by requiring 12h alignment and volume confirmation, reducing overtrading while capturing structured moves in both bull and bear markets.
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
    
    # Get 1d data for Camarilla levels and 12h for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 5 or len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using previous 1d bar's OHLC
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_1d_high - prev_1d_low
    r1 = prev_1d_close + 1.1 * camarilla_range / 12
    s1 = prev_1d_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed as they're based on completed 1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 12h EMA20 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_20_12h = close_12h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h EMA warmup, volume MA warmup
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend alignment
        trend_12h_uptrend = close[i] > ema_20_12h_aligned[i]
        trend_12h_downtrend = close[i] < ema_20_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 12h uptrend + volume spike
            long_signal = (close[i] > r1_aligned[i]) and trend_12h_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 12h downtrend + volume spike
            short_signal = (close[i] < s1_aligned[i]) and trend_12h_downtrend and volume_spike[i]
            
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
            # Exit: price breaks below S1 OR 12h trend turns down
            if (close[i] < s1_aligned[i] or not trend_12h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR 12h trend turns up
            if (close[i] > r1_aligned[i] or not trend_12h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA20_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0