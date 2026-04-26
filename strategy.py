#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike_Session_v1
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume spike confirmation, traded only during 08-20 UTC session, produce high-quality trades with low frequency. The 4h trend filter ensures alignment with medium-term momentum, while 1d volume spike confirms institutional interest. Session filter reduces noise from low-liquidity off-hours. Target: 60-120 total trades over 4 years (15-30/year).
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
    
    # Pre-compute session filter (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for HTF trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])  # previous 1d close
    
    camarilla_range = high_1d - low_1d
    r1 = close_1d_prev + 1.1 * camarilla_range / 12
    s1 = close_1d_prev - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend filter (EMA50)
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # 1d volume confirmation (volume > 2.0x 20-period average)
        volume_spike = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume spike
        if uptrend and volume_spike and breakout_r1:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short logic: breakout below S1 in downtrend with volume spike
        elif downtrend and volume_spike and breakout_s1:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike_Session_v1"
timeframe = "1h"
leverage = 1.0