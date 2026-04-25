#!/usr/bin/env python3
"""
12h Donchian20 Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian(20) breakouts from 12h chart with 1w EMA50 trend filter, volume confirmation,
and chop regime filter capture sustained momentum across market cycles. Uses 12h timeframe
to reduce trade frequency (target: 12-37/year) while capturing multi-day trends. Volume spike
(>2.0x 20-period MA) confirms institutional interest. Chop filter (CHOP < 38.2) ensures
trending regime. Discrete sizing (0.25) minimizes fee churn. Works in bull (breakouts) and
bear (breakdowns with trend filter) via symmetric long/short logic.
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
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss and volatility
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    if len(close) >= 14:
        atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
        hh = pd.Series(high).rolling(window=14, min_periods=14).max()
        ll = pd.Series(low).rolling(window=14, min_periods=14).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        chop_values = chop.values
    else:
        chop_values = np.full(n, 50.0)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_1w, ATR, Donchian, and volume MA to propagate
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_values[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_1w = ema_50_1w_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        chop = chop_values[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian upper AND uptrend (price > 1w EMA50) AND volume spike AND trending regime
            long_condition = (curr_close > upper) and (curr_close > ema50_1w) and volume_spike and trending_regime
            # Short: price breaks below Donchian lower AND downtrend (price < 1w EMA50) AND volume spike AND trending regime
            short_condition = (curr_close < lower) and (curr_close < ema50_1w) and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below Donchian lower (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above Donchian upper (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0