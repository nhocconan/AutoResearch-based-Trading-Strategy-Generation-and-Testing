#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter
Hypothesis: Daily Camarilla R1/S1 breakouts with weekly EMA50 trend filter and choppiness regime (CHOP<61.8 = trending) to avoid whipsaws in ranging markets. Uses tighter R1/S1 levels for earlier entry. Volume confirmation ensures institutional participation. Fixed size 0.25 to limit trades. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Previous weekly bar's OHLC for Camarilla levels (R1/S1 = breakout levels)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_vals = df_1w['close'].values
    
    # Calculate Camarilla levels: R1, S1 (breakout levels)
    rng = high_1w - low_1w
    camarilla_r1 = close_1w_vals + (rng * 1.1 / 12)   # R1 level
    camarilla_s1 = close_1w_vals - (rng * 1.1 / 12)   # S1 level
    
    # Align Camarilla levels to 1d timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w_vals)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    # We'll use CHOP < 61.8 as our regime filter (avoid strong ranging)
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * np.sqrt(14) / (high14 - low14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((high14 - low14) == 0, 100, chop)
    chop = np.where(np.isnan(chop), 100, chop)
    
    # Volume spike: volume > 70th percentile of 50-period lookback (volume confirmation)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_spike = volume > vol_percentile_70
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA, 50 for volume percentile, 14 for CHOP)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_percentile_70[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        size = fixed_size
        
        # Regime filter: avoid strong ranging markets (CHOP >= 61.8)
        regime_filter = chop_val < 61.8
        
        # Entry conditions: breakout of Camarilla R1/S1 with volume spike AND aligned with weekly EMA50 trend AND favorable regime
        long_entry = (close_val > camarilla_r1_val) and vol_spike and (close_val > ema_50_val) and regime_filter
        short_entry = (close_val < camarilla_s1_val) and vol_spike and (close_val < ema_50_val) and regime_filter
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter"
timeframe = "1d"
leverage = 1.0