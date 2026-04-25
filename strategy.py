#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: 12h Donchian breakouts capture medium-term momentum in both bull and bear markets.
1d EMA50 provides higher-timeframe trend alignment. Volume spike confirms institutional interest.
Choppiness filter (CHOP > 61.8) avoids whipsaws in ranging markets. Target: 12-37 trades/year
to minimize fee drag while maintaining edge in volatile crypto markets.
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
    
    # Get 1d data for HTF indicators (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate ATR(14) for stoploss and chop filter
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
        
        # Choppiness Index: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
        # CHOP = 100 * log10(sum(ATR(14)) / (max(highest_high) - min(lowest_low))) / log10(14)
        atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
        hh_14 = pd.Series(highest_high).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(lowest_low).rolling(window=14, min_periods=14).min().values
        range_14 = hh_14 - ll_14
        chop = np.zeros(n)
        for i in range(n):
            if atr_sum[i] > 0 and range_14[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / range_14[i]) / np.log10(14)
            else:
                chop[i] = 50.0  # neutral
    else:
        atr = np.full(n, 0.0)
        chop = np.full(n, 50.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, ATR, EMA, and chop to propagate
    start_idx = max(donchian_period, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_1d = ema_50_1d_aligned[i]
        upper_donchian = highest_high[i]
        lower_donchian = lowest_low[i]
        atr_val = atr[i]
        chop_val = chop[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Chop filter: only trade when market is trending (CHOP < 61.8)
        chop_filter = chop_val < 61.8
        
        if position == 0:
            # Long: price breaks above upper Donchian AND uptrend (price > 1d EMA50) AND volume spike AND trending market
            long_condition = (curr_close > upper_donchian) and (curr_close > ema50_1d) and volume_spike and chop_filter
            # Short: price breaks below lower Donchian AND downtrend (price < 1d EMA50) AND volume spike AND trending market
            short_condition = (curr_close < lower_donchian) and (curr_close < ema50_1d) and volume_spike and chop_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or price breaks below lower Donchian (reversal signal)
            if curr_close <= entry_price - 2.5 * atr_val or curr_close < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or price breaks above upper Donchian (reversal signal)
            if curr_close >= entry_price + 2.5 * atr_val or curr_close > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0