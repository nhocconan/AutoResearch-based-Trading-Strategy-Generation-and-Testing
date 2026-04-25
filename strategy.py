#!/usr/bin/env python3
"""
1h EMA Pullback with 4h Trend and Volume Spike
Hypothesis: In strong 4h trends (price > EMA50), 1h pullbacks to EMA21 with volume spikes offer high-probability entries.
Works in bull markets (buy pullbacks in uptrends) and bear markets (sell rallies in downtrends).
Uses 4h for trend filter, 1h for entry timing and volume confirmation. Targets 60-150 total trades over 4 years (15-37/year).
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
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 4h close for trend
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h EMA21 for pullback entries
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 20-period volume MA for 1h volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Session filter: 08-20 UTC (already datetime64[ms] index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA21 and volume MA
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_21[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        ema_trend_4h = ema_50_4h_aligned[i]
        ema_21_val = ema_21[i]
        vol_ma_20_val = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current 1h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_20_val
        
        if position == 0:
            # Look for entry signals
            # Long: 4h uptrend (price > EMA50) AND price pulls back to EMA21 AND volume spike
            long_entry = (curr_close > ema_trend_4h and 
                         curr_low <= ema_21_val and  # pulled back to or below EMA21
                         volume_confirm)
            # Short: 4h downtrend (price < EMA50) AND price rallies to EMA21 AND volume spike
            short_entry = (curr_close < ema_trend_4h and 
                          curr_high >= ema_21_val and  # rallied to or above EMA21
                          volume_confirm)
            
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
            # Exit: price breaks below EMA21 OR 4h trend reverses
            if curr_close < ema_21_val or curr_close < ema_trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price breaks above EMA21 OR 4h trend reverses
            if curr_close > ema_21_val or curr_close > ema_trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA_Pullback_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0