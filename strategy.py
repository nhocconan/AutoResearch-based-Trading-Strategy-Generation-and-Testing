#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_PriceAction
Hypothesis: Breakouts at 1d Camarilla R1/S1 levels with volume confirmation and 1d EMA34 trend alignment capture directional moves. Uses price action (close > open) to avoid false breakouts in sideways markets. Designed for low trade frequency (<30/year) to minimize fee drag while maintaining edge in both bull and bear markets by following 1d trend.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_PriceAction"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    range_1d = high_1d - low_1d
    s1 = close_1d - (range_1d * 1.08333)
    r1 = close_1d + (range_1d * 1.08333)
    
    # Align to 12h timeframe (wait for 1d bar to close)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # 1d trend filter: EMA 34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.8x 20-period average (tight to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.8
    
    # Price action filter: close > open (bullish candle) or close < open (bearish candle)
    bullish_candle = close > prices['open'].values
    bearish_candle = close < prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema_34_1d_aligned[i]
        is_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume confirmation + 1d uptrend + bullish candle
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend and 
                bullish_candle[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1 + volume confirmation + 1d downtrend + bearish candle
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend and 
                  bearish_candle[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S1 (opposite side)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R1 (opposite side)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals