#!/usr/bin/env python3
"""
4h Volume Spike + EMA21 Pullback with 1d Trend Filter
Hypothesis: In strong trends (defined by 1d EMA34), price pulls back to the 4h EMA21 during low volatility periods. 
A volume spike confirms renewed institutional participation at the pullback, providing a high-probability entry 
in the direction of the higher timeframe trend. This strategy works in both bull and bear markets by only taking 
trend-aligned pullbacks, avoiding counter-trend trades that fail in ranging conditions. Designed for low trade 
frequency (15-35/year) with clear entry/exit rules to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d EMA34, 4h EMA21, and volume MA
    start_idx = max(34, 21, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_21[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        ema_trend = ema_34_1d_aligned[i]
        ema_pullback = ema_21[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price pulls back to EMA21 from above AND volume spike AND 1d EMA34 uptrend
            long_entry = (curr_close >= ema_pullback * 0.998) and (curr_close <= ema_pullback * 1.002) and vol_spike and (ema_trend > ema_34_1d_aligned[i-1])
            # Short: price pulls back to EMA21 from below AND volume spike AND 1d EMA34 downtrend
            short_entry = (curr_close <= ema_pullback * 1.002) and (curr_close >= ema_pullback * 0.998) and vol_spike and (ema_trend < ema_34_1d_aligned[i-1])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below EMA21 OR 1d trend turns down
            if (curr_close < ema_pullback * 0.995) or (ema_trend < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above EMA21 OR 1d trend turns up
            if (curr_close > ema_pullback * 1.005) or (ema_trend > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_EMA21_Pullback_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0