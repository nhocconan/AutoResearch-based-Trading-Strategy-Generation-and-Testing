#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation (>2.0x 20-bar avg). 
Enters long when price breaks above H3 in 1d uptrend, short when breaks below L3 in 1d downtrend. 
Uses discrete sizing (0.25) to limit fee churn. Designed for 4h timeframe with ~20-50 trades/year, 
works in bull/bear by following 1d trend filter and requiring volume spikes to avoid false breakouts.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Range = high - low
    daily_range = high_1d - low_1d
    
    # Camarilla levels
    # H3 = close + (high - low) * 1.1 / 4
    # L3 = close - (high - low) * 1.1 / 4
    H3 = close_1d + (daily_range * 1.1 / 4)
    L3 = close_1d - (daily_range * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough data for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 in 1d uptrend with volume confirmation
            bullish_setup = (close[i] > H3_aligned[i]) and (close_1d[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: price breaks below L3 in 1d downtrend with volume confirmation
            bearish_setup = (close[i] < L3_aligned[i]) and (close_1d[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below L3 OR trend turns down
            if (close[i] < L3_aligned[i]) or (close_1d[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above H3 OR trend turns up
            if (close[i] > H3_aligned[i]) or (close_1d[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0