#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter combined with Donchian(20) breakout and volume confirmation.
# The Choppiness Index (CHOP) identifies market regime: CHOP > 61.8 = ranging (mean revert),
# CHOP < 38.2 = trending (trend follow). This filter avoids whipsaws in strong trends and false breakouts in ranges.
# In trending regimes (CHOP < 38.2), we take Donchian breakouts in the direction of the 1-day EMA(50) trend.
# Volume > 1.5x the 20-period average confirms institutional participation.
# Exit occurs when price returns to the 1-day EMA(50) or breaks the opposite Donchian band.
# This combination aims for 20-40 trades per year per symbol (80-160 total over 4 years), staying within the optimal range to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter and chop calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Choppiness Index (14 periods) on 1d
    chop_len = 14
    if len(df_1d) < chop_len:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
    tr2 = np.abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = np.abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of true ranges over chop_len periods
    tr_sum = pd.Series(tr).rolling(window=chop_len, min_periods=chop_len).sum().values
    
    # Highest high and lowest low over chop_len periods
    hh = pd.Series(df_1d['high']).rolling(window=chop_len, min_periods=chop_len).max().values
    ll = pd.Series(df_1d['low']).rolling(window=chop_len, min_periods=chop_len).min().values
    
    # Chop formula: 100 * log10(tr_sum / (hh - ll)) / log10(chop_len)
    # Avoid division by zero
    hh_ll = hh - ll
    hh_ll[hh_ll == 0] = 1e-10
    chop = 100 * np.log10(tr_sum / hh_ll) / np.log10(chop_len)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channel (20 periods) on 4h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20, chop_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + above 1d EMA + volume + trending regime
            if (close[i] > dc_upper[i] and 
                above_ema and 
                volume_confirmed and
                trending_regime):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + below 1d EMA + volume + trending regime
            elif (close[i] < dc_lower[i] and 
                  below_ema and 
                  volume_confirmed and
                  trending_regime):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 1d EMA or breaks below Donchian lower
            if close[i] < ema_1d_aligned[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 1d EMA or breaks above Donchian upper
            if close[i] > ema_1d_aligned[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_Donchian_Volume_v1"
timeframe = "4h"
leverage = 1.0