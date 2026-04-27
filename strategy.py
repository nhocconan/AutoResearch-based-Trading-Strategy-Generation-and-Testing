#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND price > 4h EMA20 AND volume spike (>1.5x avg).
Short when price breaks below Camarilla S1 AND price < 4h EMA20 AND volume spike.
Uses 4h for signal direction, 1h only for entry timing to avoid overtrading.
Session filter (08-20 UTC) reduces noise. Target: 15-35 trades/year.
Works in bull markets (breakouts with 4h uptrend) and bear markets (breakdowns with 4h downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior 4h bar (to avoid look-ahead)
    df_4h = get_htf_data(prices, '4h')
    # Prior 4h OHLC (shifted by 1 to avoid look-ahead)
    prev_close = pd.Series(df_4h['close'].values).shift(1)
    prev_high = pd.Series(df_4h['high'].values).shift(1)
    prev_low = pd.Series(df_4h['low'].values).shift(1)
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h (4 bars per 4h)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1.values)
    
    # 4h EMA20 trend filter
    ema_20 = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # Volume spike: current volume > 1.5 * 24-period average (6h)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.20  # 20% position size
    
    # Warmup: need enough for 4h prior bar (2), 4h EMA20 (~20), volume avg (24)
    start_idx = max(2, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_20_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla breakout with 4h EMA20 alignment and volume spike
            # Long: Close > Camarilla R1 AND price > 4h EMA20 AND volume spike
            # Short: Close < Camarilla S1 AND price < 4h EMA20 AND volume spike
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S1 OR loses 4h EMA20 alignment
            if close_val < s1_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R1 OR loses 4h EMA20 alignment
            if close_val > r1_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0