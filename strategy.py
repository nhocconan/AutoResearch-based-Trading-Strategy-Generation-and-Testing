#!/usr/bin/env python3
"""
6h Weekly Donchian(20) Breakout + Daily EMA34 Trend Filter + Volume Spike
Hypothesis: Weekly Donchian channels capture major structural support/resistance. 
Breakouts aligned with daily EMA34 trend and volume confirmation filter noise. 
Works in bull/bear via discrete sizing (0.25) and trend filter. 
Weekly HTF reduces whipsaw vs daily timeframe.
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
    
    # Load weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Donchian(20) - highest high/lowest low of past 20 weekly bars
    highest_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    
    # Align weekly Donchian to 6h timeframe (wait for weekly bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Load daily data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly Donchian warmup (20) and daily EMA (34)
    start_idx = max(50, 21)  # weekly Donchian needs 20, daily EMA needs 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals - require: Weekly Donchian breakout + daily EMA trend + volume spike
            # Long: price breaks above weekly Donchian high AND close > daily EMA34 AND volume spike
            long_entry = (curr_high > donchian_high_aligned[i]) and (curr_close > ema_1d_aligned[i]) and vol_spike
            # Short: price breaks below weekly Donchian low AND close < daily EMA34 AND volume spike
            short_entry = (curr_low < donchian_low_aligned[i]) and (curr_close < ema_1d_aligned[i]) and vol_spike
            
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
            # Exit: price falls back below weekly Donchian low (mean reversion) OR loss of daily EMA trend
            if (curr_low < donchian_low_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises back above weekly Donchian high (mean reversion) OR loss of daily EMA trend
            if (curr_high > donchian_high_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0