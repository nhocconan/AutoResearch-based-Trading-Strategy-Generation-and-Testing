#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R4/S4 Breakout with 4h EMA50 Trend and Volume Spike + Session Filter
- Camarilla R4/S4 levels from 1d act as extreme breakout levels (outermost) for strong momentum moves
- 4h EMA50 defines the trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 2.0x 20-period average) filters weak breakouts
- Session filter (08-20 UTC) avoids low-liquidity Asian session noise
- Uses outermost Camarilla levels (R4/S4) for higher probability, lower frequency breakouts
- Designed for 1h timeframe with tight entry conditions to target 15-30 trades/year
- Uses proven Camarilla structure with trend and volume filters to avoid overtrading
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4/S4 = C ± (H-L)*1.1/2 (outermost levels)
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 1h timeframe (use previous day's levels for breakout)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # need 1d pivots, 4h EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 AND above 4h EMA50 AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S4 AND below 4h EMA50 AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to Camarilla H3/L3 levels OR crosses 4h EMA50
            exit_signal = False
            # Calculate H3/L3 for exit (inner levels)
            camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
            camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            if position == 1:
                # Exit long when price < H3 OR < 4h EMA50
                if close[i] < camarilla_h3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > L3 OR > 4h EMA50
                if close[i] > camarilla_l3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R4S4_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0