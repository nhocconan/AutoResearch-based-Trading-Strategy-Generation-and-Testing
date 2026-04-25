#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian channel breakouts on 12h timeframe capture significant momentum moves.
Trend filter uses 1w EMA50 to ensure alignment with major trend. Volume spike confirms
breakout strength. Chop regime filter (choppiness index > 61.8) avoids whipsaws in ranging markets.
Designed for 12h timeframe with tight entry conditions to achieve 12-37 trades/year.
Works in bull (breakouts above upper channel in uptrend) and bear
(breakouts below lower channel in downtrend). Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 1w data for EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR14
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr14)/log10(hh14-ll14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = np.zeros_like(close_1d)
    mask = (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(atr_14))
    chop[mask] = 100 * np.log10(np.sum(atr_14[mask])) / (np.log10(14) * np.log10(range_14[mask]))
    chop = np.where(~mask, 50.0, chop)  # default to neutral when invalid
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels on 12h (20-period)
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, volume MA, and chop
    start_idx = max(100, donchian_window, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1w_aligned[i]
        chop_value = chop_aligned[i]
        vol_spike = volume_spike[i]
        upper_chan = upper_channel[i]
        lower_chan = lower_channel[i]
        
        # Regime filter: only trade when chop > 61.8 (ranging market) for mean reversion
        # or chop < 38.2 (trending market) for trend following
        # We'll use chop < 38.2 for trend following breakouts
        trending_regime = chop_value < 38.2
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel AND volume spike AND trending regime AND price > EMA (uptrend)
            long_entry = (curr_high > upper_chan) and vol_spike and trending_regime and (curr_close > ema_trend)
            # Short: price breaks below lower Donchian channel AND volume spike AND trending regime AND price < EMA (downtrend)
            short_entry = (curr_low < lower_chan) and vol_spike and trending_regime and (curr_close < ema_trend)
            
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
            # Exit: price crosses below lower Donchian channel OR price crosses below EMA (trend change) OR chop > 61.8 (choppy market)
            if (curr_low < lower_chan) or (curr_close < ema_trend) or (chop_value > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian channel OR price crosses above EMA (trend change) OR chop > 61.8 (choppy market)
            if (curr_high > upper_chan) or (curr_close > ema_trend) or (chop_value > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0