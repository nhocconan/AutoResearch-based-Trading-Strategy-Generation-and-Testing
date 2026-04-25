#!/usr/bin/env python3
"""
1h_VolumeSpike_TrendPullback_v1
Hypothesis: On 1h timeframe, enter long when price pulls back to EMA21 during uptrend (4h EMA50) with volume spike (>2x 20-bar avg), short when price rallies to EMA21 during downtrend with volume spike. Uses 4h for trend direction and 1h for precise entry timing. Targets 15-35 trades/year by requiring confluence of trend, pullback to EMA21, and volume spike. Session filter (08-20 UTC) reduces noise. Position size 0.20.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA50 trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMA21 for pullback entries (calculated on LTF)
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 4h EMA50 (50) and 1h EMA21 (21)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Pullback to EMA21 conditions
        pullback_long = (curr_low <= ema_21[i] * 1.002) and (curr_high >= ema_21[i] * 0.998)  # within 0.2% of EMA21
        pullback_short = (curr_high >= ema_21[i] * 0.998) and (curr_low <= ema_21[i] * 1.002)  # same condition
        
        if position == 0:
            # Look for entry signals: pullback to EMA21 with trend alignment and volume spike
            long_entry = pullback_long and uptrend and volume_confirm[i]
            short_entry = pullback_short and downtrend and volume_confirm[i]
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks above EMA21 (trend exhaustion) or trend changes
            if curr_close > ema_21[i] * 1.01 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks below EMA21 (trend exhaustion) or trend changes
            if curr_close < ema_21[i] * 0.99 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_TrendPullback_v1"
timeframe = "1h"
leverage = 1.0