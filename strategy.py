#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: On 4h timeframe, use Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume confirmation. 
Go long when price breaks above R1 with bullish 12h trend (close > 12h EMA50) and volume spike (>1.5x 20-period average). 
Go short when price breaks below S1 with bearish 12h trend (close < 12h EMA50) and volume spike. 
Exit on opposite Camarilla level touch (S1 for longs, R1 for shorts) or trend reversal. 
Designed for 20-50 trades/year on 4h by requiring multi-timeframe alignment and volume confirmation, 
reducing fee drag while capturing strong trending moves in both bull and bear markets.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get daily data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    # R2 = close + 0.5*(high-low), R1 = close + 0.25*(high-low)
    # S1 = close - 0.25*(high-low), S2 = close - 0.5*(high-low), 
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ranges
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    R1 = close_1d + 0.25 * range_1d
    S1 = close_1d - 0.25 * range_1d
    
    # Align Camarilla levels to 4h timeframe (1-day delay for previous day's data)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1, additional_delay_bars=1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1, additional_delay_bars=1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 warmup + volume MA warmup + Camarilla data availability
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend alignment
        trend_12h_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_12h_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 12h uptrend + volume spike
            long_signal = (close[i] > R1_aligned[i]) and trend_12h_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 12h downtrend + volume spike
            short_signal = (close[i] < S1_aligned[i]) and trend_12h_downtrend and volume_spike[i]
            
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
            # Exit: price touches S1 OR 12h trend turns down
            if (close[i] <= S1_aligned[i] or not trend_12h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R1 OR 12h trend turns up
            if (close[i] >= R1_aligned[i] or not trend_12h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0