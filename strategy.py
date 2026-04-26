#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: On 1h timeframe, use Camarilla R1/S1 levels from 1d for breakout entries, filtered by 4h trend direction (close > EMA20) and 1d volume spike (>2.0x 20-period average). Enter long when price breaks above R1 with 4h uptrend and 1d volume spike. Enter short when price breaks below S1 with 4h downtrend and 1d volume spike. Uses discrete position size 0.20 to balance capture and drawdown. Designed for 15-37 trades/year on 1h by requiring 4h alignment and 1d volume confirmation, reducing overtrading while capturing structured moves in both bull and bear markets.
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
    
    # Get 1d data for Camarilla levels and volume, 4h for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    if len(df_1d) < 5 or len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_1d_high - prev_1d_low
    r1 = prev_1d_close + 1.1 * camarilla_range / 12
    s1 = prev_1d_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe (no additional delay needed as they're based on completed 1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h EMA20 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d volume confirmation: volume > 2.0x 20-period average
    volume_1d_series = pd.Series(df_1d['volume'].values)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values / np.maximum(volume_ma_1d, 1e-10) > 2.0
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA warmup, volume MA warmup
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend alignment
        trend_4h_uptrend = close[i] > ema_20_4h_aligned[i]
        trend_4h_downtrend = close[i] < ema_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + 1d volume spike
            long_signal = (close[i] > r1_aligned[i]) and trend_4h_uptrend and volume_spike_1d_aligned[i]
            
            # Short: price breaks below S1 + 4h downtrend + 1d volume spike
            short_signal = (close[i] < s1_aligned[i]) and trend_4h_downtrend and volume_spike_1d_aligned[i]
            
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
            # Exit: price breaks below S1 OR 4h trend turns down
            if (close[i] < s1_aligned[i] or not trend_4h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 OR 4h trend turns up
            if (close[i] > r1_aligned[i] or not trend_4h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0