#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and ATR Trend Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirmation
filters false breakouts, while ATR-based trend filter ensures alignment with medium-term
direction. Works in bull markets (long breakouts above upper band) and bear markets
(short breakouts below lower band). 4h timeframe targets 20-50 trades/year by requiring
volume spike (>2x average) and ATR trend filter, reducing overtrading.
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
    
    # Load 1d data ONCE before loop for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) on 1d for trend strength
    atr_period = 14
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Donchian channels (20-period) on 4h
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # Donchian period, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price above/below EMA50 with ATR buffer
        uptrend = curr_close > (ema_50_1d_aligned[i] + 0.5 * atr_1d_aligned[i])
        downtrend = curr_close < (ema_50_1d_aligned[i] - 0.5 * atr_1d_aligned[i])
        
        if position == 0:
            # Look for entry signals
            # Long: break above upper channel AND uptrend AND volume spike
            long_entry = (curr_close > upper_channel[i]) and uptrend and vol_spike
            # Short: break below lower channel AND downtrend AND volume spike
            short_entry = (curr_close < lower_channel[i]) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below lower channel OR loss of uptrend
            if (curr_close < lower_channel[i]) or (curr_close < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above upper channel OR loss of downtrend
            if (curr_close > upper_channel[i]) or (curr_close > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ATRTrend"
timeframe = "4h"
leverage = 1.0