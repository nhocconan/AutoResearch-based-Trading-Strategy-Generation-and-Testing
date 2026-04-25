#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Regime Filter
Hypothesis: Donchian channel breakouts capture institutional participation, with 1d EMA34 providing trend filter, volume spike confirming strength, and chop regime ensuring trending market. Works in bull via buying upper band breakouts, bear via selling lower band breakdowns. Target: 20-50 trades/year on 4h.
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
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and chop regime
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate chop regime using ATR-based method (similar to choppiness index)
    if len(close) >= 14:
        # Sum of true ranges over 14 periods
        sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        # Max high - min low over 14 periods
        max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
        range_14 = max_high_14 - min_low_14
        # Chop value: log10(sum_tr_14 / range_14) * sqrt(14) / log10(14)
        # Normalize to 0-100 scale where >61.8 = choppy, <38.2 = trending
        chop_raw = np.where(range_14 > 0, np.log10(sum_tr_14 / range_14) * np.sqrt(14) / np.log10(14), 50)
        chop_value = chop_raw * 100  # Scale to 0-100
        trending_regime = chop_value < 38.2  # Trending market
    else:
        trending_regime = np.ones(n, dtype=bool)
    
    # Calculate Donchian channels (20-period)
    if len(close) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # Pre-compute volume spike filter
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > 2.0 * vol_ma_20
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_1d_aligned[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        atr_val = atr[i]
        is_trending = trending_regime[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper band AND uptrend AND trending regime AND volume spike
            long_condition = curr_high > upper_band and curr_close > ema_34 and is_trending and vol_spike
            # Short: break below lower band AND downtrend AND trending regime AND volume spike
            short_condition = curr_low < lower_band and curr_close < ema_34 and is_trending and vol_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below EMA34 or regime changes to choppy
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < ema_34 or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above EMA34 or regime changes to choppy
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > ema_34 or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0