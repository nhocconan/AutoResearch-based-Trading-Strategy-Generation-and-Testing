#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0

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
    
    # 1. Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    # 2. Load 1d data ONCE for Camarilla levels and volume spike
    df_1d = get_htf_data(prices, '1d')
    
    # 3. 4h EMA34 for trend filter
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 4. Calculate daily high/low/close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 5. Camarilla levels: R1, S1 (inner levels for tighter entries)
    hl_range = high_1d - low_1d
    r1 = close_1d + hl_range * 1.09
    s1 = close_1d - hl_range * 1.09
    
    # 6. Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 7. 1d volume spike filter: current volume > 2.0 * 20-period EMA
    vol_ema20_1d = pd.Series(df_1d['volume']).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_spike_1d = df_1d['volume'].values > vol_ema20_1d * 2.0
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 8. Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # 9. Fixed position size to avoid churn
    position_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema34_4h = close[i] > ema34_4h_aligned[i]
        price_below_ema34_4h = close[i] < ema34_4h_aligned[i]
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        vol_spike = volume_spike_1d_aligned[i] > 0.5
        in_session = session_filter[i]
        
        if position == 0:
            # Long: Price breaks above R1 + above 4h EMA34 + volume spike + session
            if breakout_long and price_above_ema34_4h and vol_spike and in_session:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S1 + below 4h EMA34 + volume spike + session
            elif breakout_short and price_below_ema34_4h and vol_spike and in_session:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S1 OR trend reverses
                if close[i] < s1_aligned[i] or close[i] < ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R1 OR trend reverses
                if close[i] > r1_aligned[i] or close[i] > ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals