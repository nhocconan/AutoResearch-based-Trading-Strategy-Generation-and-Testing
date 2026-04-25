#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike
Hypothesis: 1-hour Camarilla R1/S1 breakout with 4-hour trend filter and 1-day volume spike confirmation.
Targets 15-30 trades/year by requiring: 1) price breaks 4h R1/S1 levels, 2) aligned with 4h EMA20 trend,
3) confirmed by 1d volume > 1.5x 20-day average volume. Uses discrete sizing (0.20) to minimize fee drag.
Works in bull markets via trend-following breaks and in bear markets via volume-confirmed reversals at Camarilla levels.
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
    
    # 4h data for Camarilla pivots and EMA20 (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # Align 4h levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # 4h EMA20 trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d volume for spike confirmation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Start index: need enough for 4h previous data (1) + 4h EMA20 (20) + 1d volume MA (20)
    start_idx = max(20, 20) + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        
        # Trend filter: price relative to 4h EMA20
        uptrend = curr_close > ema_20_4h_aligned[i]
        downtrend = curr_close < ema_20_4h_aligned[i]
        
        # Volume confirmation: current 1h volume > 1.5x 20-day average 1d volume
        vol_spike = curr_vol > (1.5 * vol_ma_20_aligned[i])
        
        if position == 0:
            # Look for entry signals with trend alignment and volume confirmation
            # Long breakout: price breaks above R1 with uptrend and volume spike
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and vol_spike
            # Short breakout: price breaks below S1 with downtrend and volume spike
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and vol_spike
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below S1 (mean reversion) or trend changes to downtrend
            if curr_close < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit if price breaks above R1 (mean reversion) or trend changes to uptrend
            if curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0