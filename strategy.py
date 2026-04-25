#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Daily Donchian breakouts capture intermediate-term trends in both bull and bear markets.
Filtered by weekly EMA50 trend direction, volume confirmation, and choppiness regime to avoid whipsaws.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 15-25 trades/year on 1d timeframe.
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
    
    # Get 1d data for Donchian calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d (based on previous 20 days)
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    # Using previous values to avoid look-ahead
    high_series = pd.Series(df_1d['high'].values)
    low_series = pd.Series(df_1d['low'].values)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate EMA50 on 1w close for trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate choppiness index regime filter (14-period)
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # We'll use CHOP < 50 as trending regime filter to avoid whipsaws in sideways markets
    tr_range = pd.Series(high - low).rolling(window=14, min_periods=14).max().values - \
               pd.Series(high - low).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(tr_range).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(tr_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (atr_14 * 14)) / np.log10(14)
    chop_regime = chop < 50  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, volume MA, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        trending = chop_regime[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Upper AND volume spike AND price > EMA (uptrend) AND trending regime
            long_entry = (curr_close > upper) and vol_spike and (curr_close > ema_trend) and trending
            # Short: price breaks below Lower AND volume spike AND price < EMA (downtrend) AND trending regime
            short_entry = (curr_close < lower) and vol_spike and (curr_close < ema_trend) and trending
            
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
            # Exit: price crosses below Lower OR price crosses below EMA
            if (curr_close < lower) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Upper OR price crosses above EMA
            if (curr_close > upper) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0