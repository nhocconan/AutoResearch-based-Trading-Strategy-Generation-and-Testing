#!/usr/bin/env python3
# 1h_Cam_R1S1_Breakout_4hTrend_Volume
# Hypothesis: 1-hour Camarilla pivot breakout with 4-hour EMA trend filter and volume confirmation
# Uses 4h EMA50 for trend direction and 1h Camarilla levels for precise entry timing
# Volume filter reduces false breakouts. Session filter (08-20 UTC) avoids low-liquidity hours
# Designed for 1h timeframe with target of 15-35 trades/year to avoid fee drag
# Works in bull markets via long breakouts and bear markets via short breakdowns

name = "1h_Cam_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels for 1h
    # Using previous bar's high, low, close
    ph = np.roll(high, 1)
    pl = np.roll(low, 1)
    pc = np.roll(close, 1)
    ph[0] = high[0]
    pl[0] = low[0]
    pc[0] = close[0]
    
    # Camarilla R1, S1 levels
    r1 = pc + 1.1 * (ph - pl) / 12
    s1 = pc - 1.1 * (ph - pl) / 12
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1[i]) or 
            np.isnan(s1[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        # Trend filter from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume + uptrend + in session
            if close[i] > r1[i] and volume_confirm and uptrend and in_session:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume + downtrend + in session
            elif close[i] < s1[i] and volume_confirm and downtrend and in_session:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price breaks below S1 or trend reversal
            if close[i] < s1[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks above R1 or trend reversal
            if close[i] > r1[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals