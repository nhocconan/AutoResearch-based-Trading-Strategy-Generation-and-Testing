#!/usr/bin/env python3
"""
1h_Volume_Weighted_Camarilla_Breakout_4hTrend_1dRegime_v1
Hypothesis: For 1h timeframe, use 4h Camarilla R1/S1 breakouts with volume confirmation (>1.5x median) and 4h trend alignment, filtered by 1d choppiness regime (CHOP > 50 = range, < 50 = trend). Only trade breakouts in trending 1d regimes with volume confirmation. Position size 0.20 to limit drawdown. Target 15-30 trades/year/symbol by requiring confluence of 4h breakout, volume spike, and 1d trend regime.
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
    
    # Get 4h data for Camarilla levels and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 4h bar
    cam_high = pd.Series(df_4h['high'].values).shift(1).values
    cam_low = pd.Series(df_4h['low'].values).shift(1).values
    cam_close = pd.Series(df_4h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        tr1 = np.abs(high_arr - low_arr)
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_arr[0] - low_arr[0]
        
        atr = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        chop = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if atr[i] > 0 and hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr[i] / (hh[i] - ll[i])) / np.log10(window)
            else:
                chop[i] = 50.0
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # Volume spike filter: volume > 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Align HTF indicators to 1h timeframe
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(20) 4h, Camarilla (need 2 bars), volume median (20), CHOP (14)
    start_idx = max(20, 2, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(vol_median[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        ema_20_4h_val = ema_20_4h_aligned[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        chop_1d_val = chop_1d_aligned[i]
        
        # Trend filter: 4h price above/below EMA20
        uptrend_4h = close_val > ema_20_4h_val
        downtrend_4h = close_val < ema_20_4h_val
        
        # Regime filter: 1d CHOP < 50 = trending regime (favor breakouts)
        trending_regime = chop_1d_val < 50.0
        
        # Volume confirmation
        volume_spike = volume_val > 1.5 * vol_median_val
        
        if position == 0:
            # Long: break above R1 with volume spike, 4h uptrend, and trending 1d regime
            long_signal = (close_val > r1_val) and \
                          volume_spike and \
                          uptrend_4h and \
                          trending_regime
            
            # Short: break below S1 with volume spike, 4h downtrend, and trending 1d regime
            short_signal = (close_val < s1_val) and \
                           volume_spike and \
                           downtrend_4h and \
                           trending_regime
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit on 4h trend reversal or close below S1 (mean reversion)
            signals[i] = 0.20
            if close_val < s1_val or not uptrend_4h:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit on 4h trend reversal or close above R1 (mean reversion)
            signals[i] = -0.20
            if close_val > r1_val or not downtrend_4h:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Volume_Weighted_Camarilla_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0