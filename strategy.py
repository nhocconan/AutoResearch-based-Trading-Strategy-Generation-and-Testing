#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian breakouts capture momentum; 1d EMA34 filters trend to avoid counter-trend whipsaws; volume spike confirms participation; chop filter avoids ranging markets. Works in bull/bear via trend filter and volatility-based position sizing. Target: 12-37 trades/year on 12h timeframe.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate Choppiness Index (14) for regime filter
    if len(close) >= 14:
        chop_sum = tr.rolling(window=14, min_periods=14).sum()
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
        chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
        chop_values = chop.values
    else:
        chop_values = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_aligned[i]
        atr_val = atr[i]
        chop_val = chop_values[i]
        
        # Donchian channels (20-period)
        if i >= 20:
            donchian_high = np.max(high[i-19:i+1])
            donchian_low = np.min(low[i-19:i+1])
        else:
            donchian_high = np.max(high[:i+1])
            donchian_low = np.min(low[:i+1])
        
        # Volume spike: current volume > 2.0 * 50-period average
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-49:i+1])
        else:
            vol_ma_50 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_50
        
        # Trend filter
        uptrend = curr_close > ema_34
        downtrend = curr_close < ema_34
        
        # Chop filter: avoid ranging markets (chop > 61.8) and extreme trending (chop < 38.2 sometimes fails)
        # We'll use chop < 61.8 to avoid strong ranging, but allow trending markets
        not_choppy = chop_val < 61.8
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND uptrend AND not choppy
            long_condition = (curr_high > donchian_high) and volume_spike and uptrend and not_choppy
            # Short: price breaks below Donchian low AND volume spike AND downtrend AND not choppy
            short_condition = (curr_low < donchian_low) and volume_spike and downtrend and not_choppy
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or trend reversal or Donchian breakout in opposite direction
            if (curr_close <= entry_price - 2.5 * atr_val or 
                not uptrend or 
                curr_low < donchian_low):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or trend reversal or Donchian breakout in opposite direction
            if (curr_close >= entry_price + 2.5 * atr_val or 
                not downtrend or 
                curr_high > donchian_high):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0