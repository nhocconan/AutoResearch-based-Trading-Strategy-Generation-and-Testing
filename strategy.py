#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d Volume Spike + 12h EMA50 Trend Filter
Hypothesis: Donchian breakouts capture strong momentum moves. Volume confirms institutional participation.
12h EMA50 trend filter ensures we only trade in direction of intermediate trend, reducing false breakouts.
Works in bull/bear via trend filter. Target: 20-50 trades/year on 4h.
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
    
    # Get HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 50 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    if n >= 20:
        highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        highest_20 = np.full(n, np.nan)
        lowest_20 = np.full(n, np.nan)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d volume spike: current 1d volume > 2.0 * 20-period average
    if len(df_1d) >= 20:
        vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
        vol_spike_1d = df_1d['volume'].values > 2.0 * vol_ma_20
    else:
        vol_spike_1d = np.full(len(df_1d), False)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate ATR(14) for stoploss
    if n >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        ema_50 = ema_50_12h_aligned[i]
        vol_spike = vol_spike_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume spike AND uptrend
            long_condition = (curr_close > upper) and vol_spike and (curr_close > ema_50)
            # Short: price breaks below lower Donchian AND volume spike AND downtrend
            short_condition = (curr_close < lower) and vol_spike and (curr_close < ema_50)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or trend reversal
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or trend reversal
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dVolumeSpike_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0