#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_VolumeSpike_1dTrend_v6
Hypothesis: Camarilla pivot levels (H3/L3) from 1d timeframe with volume spike and 1d EMA trend filter capture mean-reversion bounces in range-bound markets and breakouts in trending markets, working across bull/bear regimes. Targets 20-40 trades/year to minimize fee drag.
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
    
    # Get 1d data for Camarilla pivots and EMA trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Camarilla levels
    range_ = phigh - plow
    H3 = pclose + (range_ * 1.1 / 4)
    L3 = pclose - (range_ * 1.1 / 4)
    H4 = pclose + (range_ * 1.1 / 2)
    L4 = pclose - (range_ * 1.1 / 2)
    
    # Align to 4h timeframe
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d EMA trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or np.isnan(H4_4h[i]) or np.isnan(L4_4h[i]) or
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price touches or crosses above L3 with volume spike and above EMA (bullish bias)
            if price > L3_4h[i] and volume_spike[i] and price > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses below H3 with volume spike and below EMA (bearish bias)
            elif price < H3_4h[i] and volume_spike[i] and price < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price reaches H4 (take profit) or crosses below EMA (trend change)
            if price >= H4_4h[i] or price < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price reaches L4 (take profit) or crosses above EMA (trend change)
            if price <= L4_4h[i] or price > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_VolumeSpike_1dTrend_v6"
timeframe = "4h"
leverage = 1.0