#!/usr/bin/env python3
"""
1h Volume Spike + 4h EMA Trend + 1d Session Filter
Hypothesis: During liquid sessions (08-20 UTC), volume spikes on 1h in direction of 4h EMA(50) trend capture institutional participation. 
4h EMA provides medium-term trend filter to avoid counter-trend whipsaws, while session filter reduces noise from low-liquidity periods.
Designed for 1h timeframe targeting 15-35 trades/year. Uses discrete position sizing (0.20) to minimize fee churn.
Works in bull markets via trend continuation and in bear markets via mean-reversion from overextended moves during session hours.
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
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_4h and volume MA to propagate
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        ema50_4h = ema_50_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 2.5 * 20-period average
        volume_spike = curr_volume > 2.5 * vol_ma
        
        if position == 0:
            # Long: volume spike AND price above 4h EMA50 (uptrend)
            long_condition = volume_spike and (curr_close > ema50_4h)
            # Short: volume spike AND price below 4h EMA50 (downtrend)
            short_condition = volume_spike and (curr_close < ema50_4h)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price closes below 4h EMA50 (trend reversal)
            if curr_close < ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above 4h EMA50 (trend reversal)
            if curr_close > ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_4hEMA50_Trend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0