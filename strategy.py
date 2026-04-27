#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol
Hypothesis: Price breaking through R1/S1 levels on 1h with 4h trend and 1d volume spike captures breakout moves. 
In bull markets, R1 breakouts trigger longs; in bear markets, S1 breakdowns trigger shorts. 
Uses 4h trend filter (EMA21) and 1d volume confirmation to reduce false signals. 
Targets 15-37 trades/year on 1h to minimize fee drag while capturing directional moves.
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
    
    # Get 1h data for Camarilla pivot calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1h bar
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1h - low_1h) * 1.1 / 12
    r1_1h = close_1h + camarilla_range
    s1_1h = close_1h - camarilla_range
    
    # Align R1/S1 to 1h timeframe (use previous 1h bar's levels)
    r1_1h_aligned = align_htf_to_ltf(prices, df_1h, r1_1h)
    s1_1h_aligned = align_htf_to_ltf(prices, df_1h, s1_1h)
    
    # 4h trend filter: EMA21
    df_4h = get_htf_data(prices, '4h')
    ema21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # 1d volume confirmation: volume > 2.0 * 20-period average
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r1_1h_aligned[i]) or np.isnan(s1_1h_aligned[i]) or 
            np.isnan(ema21_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        r1 = r1_1h_aligned[i]
        s1 = s1_1h_aligned[i]
        ema_trend = ema21_4h_aligned[i]
        vol_spike_val = vol_spike_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and volume spike
            if close[i] > r1 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with downtrend and volume spike
            elif close[i] < s1 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below S1 or trend turns down
            if close[i] < s1 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above R1 or trend turns up
            if close[i] > r1 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0