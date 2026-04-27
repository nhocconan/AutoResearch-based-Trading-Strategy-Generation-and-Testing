#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_HTF
Hypothesis: Camarilla R4/S4 breakout on 6h with 1d trend filter (price > EMA34) and volume spike.
Breakouts at R4/S4 levels (stronger than R3/S3) with 1d trend alignment and volume confirmation 
capture strong momentum moves while avoiding false breakouts. R4/S4 represent 50% retracement 
levels of the daily range, making them significant support/resistance. Works in both bull and bear
markets by only trading in the direction of the 1d trend. Target: 50-150 total trades over 4 years.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (focus on R4/S4 for stronger breakouts)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Key levels: R4 and S4 for breakout entries (50% retracement levels)
    R4 = PP + range_1d * 1.1 / 2.0
    S4 = PP - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for EMA34 and volume average
    start_idx = max(100, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 1d trend with volume spike
            # Long: price breaks above R4 AND 1d trend is up (price > EMA34) AND volume spike
            # Short: price breaks below S4 AND 1d trend is down (price < EMA34) AND volume spike
            long_breakout = close_val > R4_aligned[i]
            short_breakout = close_val < S4_aligned[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_breakout and trend_up and vol_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and trend_down and vol_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below S4 (failed breakout) or volume drops
            if close_val < S4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R4 (failed breakout) or volume drops
            if close_val > R4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_HTF"
timeframe = "6h"
leverage = 1.0