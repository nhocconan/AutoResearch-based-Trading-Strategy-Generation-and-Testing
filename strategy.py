#!/usr/bin/env python3
"""
1h Camarilla Pivot H3L3 Breakout + 4h EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance on 4h timeframe.
Breakouts above H3 (resistance) or below L3 (support) with volume confirmation
and aligned with 4h EMA34 trend capture momentum. Uses 1h for precise entry timing
while 4h/1d provide signal direction. Session filter (08-20 UTC) reduces noise.
Target: 15-37 trades/year (60-150 over 4 years) with discrete sizing 0.20 to minimize fee drag.
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
    
    # Get 4h data for Camarilla pivot and EMA (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 4h
    # Based on previous 4h bar's high, low, close
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    h3 = pivot + range_val * 1.1 / 4.0  # Resistance level 3
    l3 = pivot - range_val * 1.1 / 4.0  # Support level 3
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Calculate EMA34 on 4h close for trend
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume (1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_4h_aligned[i]
        vol_spike = volume_spike[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 (resistance) AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > h3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 (support) AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < l3_level) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below H3 OR price crosses below EMA (trend change)
            if (curr_low < h3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above L3 OR price crosses above EMA (trend change)
            if (curr_high > l3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0