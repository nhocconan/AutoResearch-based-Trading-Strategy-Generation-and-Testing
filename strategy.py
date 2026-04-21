#!/usr/bin/env python3
"""
4h_1D_TrendFollow_With_Volume_Confirmation
Hypothesis: Use daily EMA as trend filter, enter on 4h EMA pullbacks with volume confirmation. Works in bull markets by buying dips in uptrend, in bear markets by selling rallies in downtrend. Low trade frequency (~30/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on daily closes
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h EMA for entry timing
    close_4h = prices['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if daily EMA not ready
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price relative to daily EMA
        price_above_daily_ema = price > ema_34_1d_aligned[i]
        price_below_daily_ema = price < ema_34_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price above daily EMA + pullback to 4h EMA + volume
            if price_above_daily_ema and price <= ema_21_4h[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA + rally to 4h EMA + volume
            elif price_below_daily_ema and price >= ema_21_4h[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4h EMA or trend changes
            if price < ema_21_4h[i] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 4h EMA or trend changes
            if price > ema_21_4h[i] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_TrendFollow_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0