#!/usr/bin/env python3
"""
1h Camarilla Pivot Breakout with 4h Trend Filter and Session Timing
Hypothesis: Camarilla H3/L3 levels from prior 1d act as key intraday support/resistance.
Breakouts with volume confirmation and aligned 4h EMA20 trend capture momentum moves.
Restricted to 08-20 UTC session to avoid low-liquidity hours. Designed for 1h timeframe
with tight entry conditions (target: 15-37 trades/year) to minimize fee drag. Uses 4h
for signal direction and 1h for entry timing precision.
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA20 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 4h close for trend
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(
        span=20, adjust=False, min_periods=20
    ).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for Camarilla pivot levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    range_ = prev_high - prev_low
    camarilla_h3 = prev_close + range_ * 1.1 / 4
    camarilla_l3 = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume spike: current volume > 2.0 * 24-period average volume (~1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and prior day data
    start_idx = max(24, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i]:
            signals[i] = 0.0
            continue
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        ema_trend = ema_20_4h_aligned[i]
        camarilla_h3 = h3_aligned[i]
        camarilla_l3 = l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla H3 AND volume spike AND price > 4h EMA20 (uptrend)
            long_entry = (curr_close > camarilla_h3) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Camarilla L3 AND volume spike AND price < 4h EMA20 (downtrend)
            short_entry = (curr_close < camarilla_l3) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below Camarilla L3 (support) OR price crosses below EMA (trend change)
            if (curr_close < camarilla_l3) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above Camarilla H3 (resistance) OR price crosses above EMA (trend change)
            if (curr_close > camarilla_h3) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA20_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0