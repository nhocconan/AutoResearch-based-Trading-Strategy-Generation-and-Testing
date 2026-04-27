#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h trend filter (price > EMA21) and volume spike.
Breakouts at R1/S1 levels with 4h trend alignment and volume confirmation capture strong moves
while avoiding false breakouts in choppy or counter-trend conditions. Uses 4h for signal direction,
1h only for entry timing to minimize fee drift. Works in both bull and bear markets by only trading
in the direction of the 4h trend.
Target: 60-150 total trades over 4 years (15-37/year) for BTC/ETH/SOL.
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
    
    # Precompute session filter (08-20 UTC) once
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA21 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 4h Camarilla pivot levels (focus on R1/S1 for breakouts)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    PP = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Key levels: R1 and S1 for breakout entries
    R1 = PP + range_4h * 1.1 / 12.0
    S1 = PP - range_4h * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.20  # 20% position size
    
    # Warmup: need enough for EMA21 and volume average
    start_idx = max(100, 21, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_21_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 4h trend with volume spike
            # Long: price breaks above R1 AND 4h trend is up (price > EMA21) AND volume spike
            # Short: price breaks below S1 AND 4h trend is down (price < EMA21) AND volume spike
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0