#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session
Hypothesis: On 1h timeframe, use Camarilla R1/S1 levels from 4h for breakout entries, filtered by 4h trend direction (close > EMA34) and volume spike (>2.0x 20-period average). Restrict trading to 08-20 UTC session to avoid low-liquidity hours. Uses discrete position size 0.20. Target: 15-37 trades/year by requiring 4h alignment, volume confirmation, and session filter. Designed to work in both bull and bear markets by capturing structured breakouts with trend alignment.
"""

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
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's OHLC)
    prev_4h_close = df_4h['close'].shift(1).values
    prev_4h_high = df_4h['high'].shift(1).values
    prev_4h_low = df_4h['low'].shift(1).values
    
    camarilla_range = prev_4h_high - prev_4h_low
    r1 = prev_4h_close + 1.1 * camarilla_range / 12
    s1 = prev_4h_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Calculate 4h EMA34 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Session filter: 08-20 UTC (using DatetimeIndex hour)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA warmup, volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Check session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # 4h trend alignment
        trend_4h_uptrend = close[i] > ema_34_4h_aligned[i]
        trend_4h_downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume spike
            long_signal = (close[i] > r1_aligned[i]) and trend_4h_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 4h downtrend + volume spike
            short_signal = (close[i] < s1_aligned[i]) and trend_4h_downtrend and volume_spike[i]
            
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0