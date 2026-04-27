#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d trend filter (price > EMA34) and volume spike.
Breakouts at R1/S1 levels with 1d trend alignment and volume confirmation capture strong moves
while avoiding false breakouts in choppy or counter-trend conditions. Works in both bull and bear
markets by only trading in the direction of the 1d trend.
Target: 75-200 total trades over 4 years (19-50/year) for BTC/ETH/SOL.
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
    
    # Calculate 1d Camarilla pivot levels (focus on R1/S1 for breakouts)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Key levels: R1 and S1 for breakout entries (tighter than R3/S3)
    R1 = PP + range_1d * 1.1 / 12.0
    S1 = PP - range_1d * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
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
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 1d trend with volume spike
            # Long: price breaks above R1 AND 1d trend is up (price > EMA34) AND volume spike
            # Short: price breaks below S1 AND 1d trend is down (price < EMA34) AND volume spike
            long_breakout = close_val > R1_aligned[i]
            short_breakout = close_val < S1_aligned[i]
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
            # Long - exit when price breaks below S1 (failed breakout) or volume drops
            if close_val < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R1 (failed breakout) or volume drops
            if close_val > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0