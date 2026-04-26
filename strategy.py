#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v17
Hypothesis: On 4h timeframe, trade Camarilla R1/S1 breakouts aligned with 1d EMA34 trend and volume spikes. 
This strategy targets 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.25) to minimize fee drag.
The 1d EMA34 acts as a regime filter: only take longs when price is above EMA34 (bullish regime) and shorts when below EMA34 (bearish regime).
Volume spike confirms institutional participation at the breakout. Works in bull/bear markets via 1d trend filter.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla levels use previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12.0
    s1 = prev_close - camarilla_range * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA warmup (34) and volume MA (20)
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_val = ema_34_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1, above 1d EMA34, volume spike
            long_signal = (close_val > r1_val) and (close_val > ema_34_val) and vol_spike
            
            # Short: price breaks below S1, below 1d EMA34, volume spike
            short_signal = (close_val < s1_val) and (close_val < ema_34_val) and vol_spike
            
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
            # Exit: price breaks below S1 OR below 1d EMA34
            if (close_val < s1_val) or (close_val < ema_34_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR above 1d EMA34
            if (close_val > r1_val) or (close_val > ema_34_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v17"
timeframe = "4h"
leverage = 1.0