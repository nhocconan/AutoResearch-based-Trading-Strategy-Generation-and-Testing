#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_ChopFilter
Hypothesis: Donchian(20) breakout on 4h with 12h EMA50 trend filter, volume confirmation, and choppiness regime filter.
Breakouts above/below Donchian(20) channels capture strong momentum moves. Trend filter ensures we only trade
in direction of 12h trend to avoid counter-trend whipsaws. Volume spike confirms breakout authenticity.
Choppiness filter avoids ranging markets (CHOP > 61.8) and only trades in trending regimes (CHOP < 38.2).
Designed for 4h timeframe with target 75-200 trades over 4 years (19-50/year). Uses discrete position sizing (0.25)
to balance return and drawdown. Works in both bull and bear markets by aligning with intermediate-term 12h trend
and avoiding false signals in ranging markets via choppiness filter.
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
    
    # Calculate ATR for stoploss (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h choppiness index (14-period)
    df_12h_chop = get_htf_data(prices, '12h')
    if len(df_12h_chop) < 14:
        return np.zeros(n)
    
    high_12h = df_12h_chop['high'].values
    low_12h = df_12h_chop['low'].values
    close_12h = df_12h_chop['close'].values
    
    # True range for 12h
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    
    # Sum of true ranges over 14 periods
    tr_sum_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index: CHOP = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_12h - ll_12h
    chop_raw = np.where(hh_ll_diff > 0, tr_sum_12h / hh_ll_diff, 1.0)
    chop_raw = np.maximum(chop_raw, 1e-10)  # prevent log of zero
    chop_12h = 100 * np.log10(chop_raw) / np.log10(14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h_chop, chop_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for ATR, Donchian, EMA50, chop, volume average
    start_idx = max(100, 20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_trend = ema_50_aligned[i]
        chop_val = chop_12h_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 12h trend with volume spike and trending regime
            # Long: price breaks above Donchian high AND 12h trend is up (close > EMA50) AND volume spike AND trending
            # Short: price breaks below Donchian low AND 12h trend is down (close < EMA50) AND volume spike AND trending
            long_breakout = close_val > donch_high
            short_breakout = close_val < donch_low
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_breakout and trend_up and vol_spike and is_trending:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and trend_down and vol_spike and is_trending:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Donchian low (failed breakout) or ATR stoploss hit
            if close_val < donch_low or close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian high (failed breakout) or ATR stoploss hit
            if close_val > donch_high or close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0