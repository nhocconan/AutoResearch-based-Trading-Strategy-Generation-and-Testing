#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime
Hypothesis: Trade 12h Camarilla R1/S1 breakouts in direction of 1d EMA34 trend with volume spike and chop regime filter.
Uses Camarilla pivot levels from daily timeframe for structure, 1d EMA34 for higher timeframe trend alignment,
volume spike on 12h for breakout conviction, and choppiness index to avoid whipsaws in ranging markets.
Designed to work in both bull and bear markets via trend filter + volume confirmation + regime filter.
Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag.
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
    
    # Get 12h data for volume MA and price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # Based on prior day's range
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_r1 = camarilla_pivot + (range_1d * 1.0 / 12)
    camarilla_s1 = camarilla_pivot - (range_1d * 1.0 / 12)
    camarilla_r2 = camarilla_pivot + (range_1d * 2.0 / 12)
    camarilla_s2 = camarilla_pivot - (range_1d * 2.0 / 12)
    camarilla_r3 = camarilla_pivot + (range_1d * 3.0 / 12)
    camarilla_s3 = camarilla_pivot - (range_1d * 3.0 / 12)
    camarilla_r4 = camarilla_pivot + (range_1d * 4.0 / 12)
    camarilla_s4 = camarilla_pivot - (range_1d * 4.0 / 12)
    
    # Align Camarilla levels to 12h timeframe (prior 1d's levels available at 1d close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 2.0x 20-period average on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    volume_series = pd.Series(volume_12h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Choppiness Index regime filter (using 12h data)
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # We'll use CHOP < 50 as our regime filter to avoid strong ranging markets
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index
    chop = np.where(
        (sum_atr_14 > 0) & (range_14 > 0),
        100 * np.log10(sum_atr_14 / range_14) / np.log10(14),
        50  # Neutral when undefined
    )
    
    # Align chop to 12h timeframe (it's already on 12h)
    chop_aligned = chop  # Already calculated on 12h data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), volume MA (20), ATR (14+14), chop (14+14)
    start_idx = max(34, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_ma[i]) or
            np.isnan(chop_aligned[i])):
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
        
        # Regime filter: avoid strong ranging markets (chop > 50)
        not_ranging = chop_aligned[i] < 50
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + price above 1d EMA34 + volume spike + not ranging
            long_breakout = close[i] > camarilla_r1_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i] and not_ranging
            
            # Short: price breaks below Camarilla S1 + price below 1d EMA34 + volume spike + not ranging
            short_breakout = close[i] < camarilla_s1_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i] and not_ranging
            
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
            # Exit: price touches Camarilla S1 OR trend turns bearish (price below EMA) OR strong ranging market
            if (close[i] < camarilla_s1_aligned[i] or not price_above_ema or chop_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 OR trend turns bullish (price above EMA) OR strong ranging market
            if (close[i] > camarilla_r1_aligned[i] or not price_below_ema or chop_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0