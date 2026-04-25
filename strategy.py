#!/usr/bin/env python3
"""
1h Volume Spike Pullback + 4h EMA50 Trend + Session Filter
Hypothesis: In trending markets (4h EMA50), 1h pullbacks to EMA20 with volume spikes offer high-probability entries. Session filter (08-20 UTC) avoids low-liquidity hours. Works in bull/bear via discrete sizing (0.20) and trend filter.
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
    
    # Load 4h data ONCE before loop for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h EMA20 for pullback entries
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 4h EMA warmup and volume MA
    start_idx = max(50, 21)  # EMA50 needs ~50, vol MA 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        in_session = session_filter[i]
        
        if position == 0:
            # Look for entry signals - require: Pullback to EMA20 + volume spike + trend + session
            # Long: price pulls back to EMA20 from above AND volume spike AND bullish 4h trend AND in session
            long_entry = (curr_low <= ema_20[i] and curr_close > ema_20[i]) and vol_spike and (curr_close > ema_4h_aligned[i]) and in_session
            # Short: price pulls back to EMA20 from below AND volume spike AND bearish 4h trend AND in session
            short_entry = (curr_high >= ema_20[i] and curr_close < ema_20[i]) and vol_spike and (curr_close < ema_4h_aligned[i]) and in_session
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below EMA20 OR loss of bullish 4h trend OR outside session
            if (curr_close < ema_20[i]) or (curr_close < ema_4h_aligned[i]) or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price breaks above EMA20 OR loss of bearish 4h trend OR outside session
            if (curr_close > ema_20[i]) or (curr_close > ema_4h_aligned[i]) or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_Pullback_EMA20_4hEMA50_Trend_SessionFilter"
timeframe = "1h"
leverage = 1.0