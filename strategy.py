#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian channel breakouts on 12h timeframe capture medium-term momentum.
Using 1d EMA34 for trend filter ensures alignment with daily trend. Volume spike confirms
institutional participation. Chop filter avoids whipsaws in ranging markets.
Designed to work in both bull and bear markets via trend-following logic.
Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years).
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 35 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h (using 12h high/low)
    # Since we're on 12h timeframe, we can calculate directly
    # But to avoid look-ahead, we use rolling window on historical data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Choppiness Index (14) on 12h to avoid ranging markets
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # only allow trades when not strongly ranging (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, ATR, and CHOP
    start_idx = max(20, 34, 14, 20)  # 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        not_choppy = chop_filter[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian AND volume spike AND price > EMA (uptrend) AND not choppy
            long_entry = (curr_close > upper_donchian) and vol_spike and (curr_close > ema_trend) and not_choppy
            # Short: price breaks below lower Donchian AND volume spike AND price < EMA (downtrend) AND not choppy
            short_entry = (curr_close < lower_donchian) and vol_spike and (curr_close < ema_trend) and not_choppy
            
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
            # Exit: price crosses below lower Donchian OR price crosses below EMA
            if (curr_close < lower_donchian) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian OR price crosses above EMA
            if (curr_close > upper_donchian) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0