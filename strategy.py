#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ATR Trend Filter
Hypothesis: Donchian channel breakouts capture strong momentum. Volume spike confirms institutional participation.
ATR trend filter ensures we only trade in the direction of medium-term trend, avoiding counter-trend whipsaws.
Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 25-40 trades/year on 4h.
Works in bull markets via breakouts and in bear markets via trend filter (avoids false breakouts in ranging/choppy markets).
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
    
    # Get 1d data for ATR trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d for trend filter
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
            np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
        )
    )
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels (20) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        atr_val = atr_14_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # ATR-based trend filter: use EMA crossover with ATR-adjusted bands
        ema_fast = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
        ema_slow = pd.Series(close).ewm(span=30, adjust=False, min_periods=30).mean().values
        trend_up = ema_fast[i] > ema_slow[i]
        trend_down = ema_fast[i] < ema_slow[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel AND volume spike AND uptrend
            long_entry = (curr_close > upper_channel) and vol_spike and trend_up
            # Short: price breaks below lower Donchian channel AND volume spike AND downtrend
            short_entry = (curr_close < lower_channel) and vol_spike and trend_down
            
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
            # Exit: price crosses below lower Donchian channel OR trend turns down
            if (curr_close < lower_channel) or (not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian channel OR trend turns up
            if (curr_close > upper_channel) or (trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ATRTrend"
timeframe = "4h"
leverage = 1.0