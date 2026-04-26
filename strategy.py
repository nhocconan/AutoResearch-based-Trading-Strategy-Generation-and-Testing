#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike
Hypothesis: On 1h timeframe, use Camarilla R1/S1 levels from 4h for breakout entries, filtered by 1d trend (close > EMA50) and volume spike (>1.8x 20-period average). Enter long when price breaks above R1 with 1d uptrend and volume spike. Enter short when price breaks below S1 with 1d downtrend and volume spike. Uses discrete position size 0.20 to minimize fee churn. Designed for 15-35 trades/year on 1h by requiring daily alignment and volume confirmation, reducing overtrading while capturing structured moves in both bull and bear markets.
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
    
    # Get 4h data for Camarilla levels and 1d for trend filter
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 5 or len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's OHLC)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_4h_close = df_4h['close'].shift(1).values
    prev_4h_high = df_4h['high'].shift(1).values
    prev_4h_low = df_4h['low'].shift(1).values
    
    camarilla_range = prev_4h_high - prev_4h_low
    r1 = prev_4h_close + 1.1 * camarilla_range / 12
    s1 = prev_4h_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA warmup, volume MA warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i]) or not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike
            long_signal = (close[i] > r1_aligned[i]) and trend_1d_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 1d downtrend + volume spike
            short_signal = (close[i] < s1_aligned[i]) and trend_1d_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S1 OR 1d trend turns down
            if (close[i] < s1_aligned[i] or not trend_1d_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 OR 1d trend turns up
            if (close[i] > r1_aligned[i] or not trend_1d_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0