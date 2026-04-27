#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Volume_Regime
Hypothesis: Daily strategy using Camarilla R3/S3 from weekly pivot for breakout entries with volume confirmation and choppiness regime filter. 
Only trade when weekly trend is established (price > weekly EMA50 for longs, < weekly EMA50 for shorts) to avoid whipsaws. 
Designed for low trade frequency (~15-25/year) with discrete position sizing (0.25) to minimize fee drag in bear markets.
Works in both bull and bear markets by following weekly trend while using daily Camarilla levels for precise entries.
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
    
    # Get weekly data for trend filter and Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly OHLC for Camarilla levels (using prior week's data)
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Camarilla levels: R3/S3 from weekly OHLC (wider bands for fewer false breakouts)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_1w + (h_1w - l_1w) * 1.1 / 4
    camarilla_s3 = c_1w - (h_1w - l_1w) * 1.1 / 4
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(c_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe (completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Choppiness regime filter: CHOP(14) < 61.8 = trending (favor breakouts)
    # Higher CHOP = more choppy, avoid breakouts in choppy markets
    tr_range = pd.Series(high - low).rolling(window=14, min_periods=14).max().values - \
               pd.Series(high - low).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(tr_range).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(atr_14 / (pd.Series(tr_range).sum())) / np.log10(14)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    chop_regime = chop < 61.8  # Trending regime favoring breakouts
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need weekly EMA50 (50) + volume avg (20) + chop (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        chop_reg = chop_regime[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with weekly trend filter, volume spike, and trending regime
            # Long: price closes above R3 AND above weekly EMA50 (weekly uptrend) AND volume spike AND trending regime
            long_condition = (close_val > r3_val) and (close_val > ema_val) and vol_conf and chop_reg
            # Short: price closes below S3 AND below weekly EMA50 (weekly downtrend) AND volume spike AND trending regime
            short_condition = (close_val < s3_val) and (close_val < ema_val) and vol_conf and chop_reg
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S3 (opposite level) OR weekly EMA50 turns bearish (price below EMA)
            if (close_val < s3_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (opposite level) OR weekly EMA50 turns bullish (price above EMA)
            if (close_val > r3_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_Pivot_Volume_Regime"
timeframe = "1d"
leverage = 1.0