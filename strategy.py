#!/usr/bin/env python3
"""
1h_VolumeSpike_Pullback_4hTrend
Hypothesis: On 1h timeframe, enter long on volume-confirmed pullbacks to 20 EMA during 4h uptrend,
and short on volume-confirmed rallies to 20 EMA during 4h downtrend. Uses session filter (08-20 UTC)
to avoid low-liquidity periods. Designed for 15-37 trades/year by requiring volume spikes (>2x average)
and strong 4h trend alignment (EMA50). Discrete position sizing (0.20) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for EMA50 trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Precompute indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA20 for pullback entries
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1h volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: 08-20 UTC (precompute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if indicators not ready
        if (np.isnan(ema_20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema20 = ema_20[i]
        vol_avg = vol_ma[i]
        ema50_4h = ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 2x average (strict to reduce trades)
        volume_spike = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: volume spike + pullback to EMA20 during 4h uptrend
            long_condition = volume_spike and (price <= ema20 * 1.005) and (price >= ema20 * 0.995) and (ema50_4h > ema20)
            
            # Short: volume spike + rally to EMA20 during 4h downtrend
            short_condition = volume_spike and (price >= ema20 * 0.995) and (price <= ema20 * 1.005) and (ema50_4h < ema20)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
                
        elif position == 1:
            # Exit long: price breaks above EMA20 or 4h trend turns down
            if price > ema20 * 1.01 or ema50_4h < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Exit short: price breaks below EMA20 or 4h trend turns up
            if price < ema20 * 0.99 or ema50_4h > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_Pullback_4hTrend"
timeframe = "1h"
leverage = 1.0