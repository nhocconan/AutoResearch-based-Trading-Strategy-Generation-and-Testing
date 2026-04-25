#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wTrend_VolumeConfirm
Hypothesis: 12h Camarilla H3/L3 breakout with 1w trend filter (price > EMA34 on 1w) and volume confirmation (>1.8x 24-bar avg). 
Enters long when price breaks above H3 with 1w uptrend and volume spike, short when breaks below L3 with 1w downtrend and volume spike.
Uses discrete sizing (0.25) to limit fee churn. Designed for 12h timeframe with ~15-35 trades/year, works in bull/bear by following 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Use previous day's OHLC for today's Camarilla levels (no look-ahead)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # H3 = Close + 1.1*(High-Low)/4
    # L3 = Close - 1.1*(High-Low)/4
    camarilla_range = high_1d - low_1d
    h3 = close_1d + 1.1 * camarilla_range / 4
    l3 = close_1d - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 12h timeframe (using previous day's close)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 1.8x 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough data for EMA34 and volume MA
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 with 1w uptrend and volume confirmation
            long_setup = (close[i] > h3_aligned[i]) and (close_1w[i] > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short: price breaks below L3 with 1w downtrend and volume confirmation
            short_setup = (close[i] < l3_aligned[i]) and (close_1w[i] < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below L3 OR 1w trend turns down
            if (close[i] < l3_aligned[i]) or (close_1w[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above H3 OR 1w trend turns up
            if (close[i] > h3_aligned[i]) or (close_1w[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0