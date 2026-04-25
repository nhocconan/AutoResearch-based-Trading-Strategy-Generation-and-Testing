#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Trade 4h Camarilla H3/L3 breakouts with 1d EMA34 trend filter and 1d volume spike (>2.0x 20-bar MA). Uses chop regime filter (CHOP > 61.8 = range, mean revert; CHOP < 38.2 = trend, trend follow). Designed to work in bull/bear via trend filter + volume confirmation + regime adaptation. Target 20-50 trades/year on 4h timeframe.
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
    
    # Get 1d data for HTF trend, volume, and chop regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA34 on 1d for HTF trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Choppiness Index on 1d for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        atr = np.zeros_like(close_arr)
        tr1 = np.abs(high_arr - low_arr)
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        chop = np.where((hh - ll) != 0, 
                        100 * np.log10(atr * window / (hh - ll)) / np.log10(window), 
                        50)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels from previous 4h bar (for 4h entry timing)
    camarilla_range = (high - low) * 1.1 / 12.0
    camarilla_H3 = close + camarilla_range * 1.25
    camarilla_L3 = close - camarilla_range * 1.25
    
    # Shift by 1 to use only completed 4h bar for Camarilla calculation (no look-ahead)
    camarilla_H3 = np.roll(camarilla_H3, 1)
    camarilla_L3 = np.roll(camarilla_L3, 1)
    camarilla_H3[0] = np.nan
    camarilla_L3[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), chop (14), and Camarilla (1)
    start_idx = max(34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(camarilla_H3[i]) or 
            np.isnan(camarilla_L3[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
        in_range = chop_1d_aligned[i] > 61.8
        in_trend = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long setup: price breaks above Camarilla H3 + above 1d EMA34 + volume spike
            # In trend: follow breakout; in range: mean revert (so short at resistance, long at support)
            long_setup = (close[i] > camarilla_H3[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i] and \
                         in_trend  # Only long breakouts in trending regime
            # Short setup: price breaks below Camarilla L3 + below 1d EMA34 + volume spike
            short_setup = (close[i] < camarilla_L3[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i] and \
                          in_trend  # Only short breakdowns in trending regime
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below Camarilla L3 OR below 1d EMA34
            if (close[i] < camarilla_L3[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Camarilla H3 OR above 1d EMA34
            if (close[i] > camarilla_H3[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0