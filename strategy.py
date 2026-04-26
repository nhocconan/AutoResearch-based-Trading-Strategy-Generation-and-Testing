#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter
Hypothesis: Trade 12h Camarilla R1/S1 breakouts aligned with 1d EMA34 trend and choppiness regime filter.
Uses 12h primary timeframe for low trade frequency (target: 12-37/year). 
Camarilla levels from prior 1d provide structure; 1d EMA34 filters trend direction; 
1d choppiness index avoids whipsaw in ranging markets. Volume confirmation ensures breakout strength.
Works in bull/bear via trend filter + chop regime filter reducing false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14) sum
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll = hh - ll
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    chop = 100 * np.log10(atr_sum / np.maximum(hh_ll, 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Camarilla levels for today (based on prior 1d OHLC)
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    camarilla_R1 = close_1d + camarilla_range
    camarilla_S1 = close_1d - camarilla_range
    
    # Align Camarilla levels to 12h timeframe (prior 1d's levels available at 00:00 UTC)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: volume > 1.8x 20-period average on 12h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), CHOP (34), volume MA (20), aligned indicators
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Regime filter: chop < 61.8 = trending (favor breakouts), chop > 61.8 = ranging (avoid)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + price above 1d EMA34 + volume spike + trending regime
            long_breakout = close[i] > camarilla_R1_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i] and trending_regime
            
            # Short: price breaks below Camarilla S1 + price below 1d EMA34 + volume spike + trending regime
            short_breakout = close[i] < camarilla_S1_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i] and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Camarilla S1 OR trend turns bearish OR chop becomes too high (range)
            if (close[i] < camarilla_S1_aligned[i] or not price_above_ema or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 OR trend turns bullish OR chop becomes too high (range)
            if (close[i] > camarilla_R1_aligned[i] or not price_below_ema or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter"
timeframe = "12h"
leverage = 1.0