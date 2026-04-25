#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and 1d volume spike (>2.0x 20-bar MA). Uses choppiness regime filter to avoid whipsaws in ranging markets. Designed for fewer trades (<50/year) to minimize fee drag while capturing strong trends in both bull and bear markets via trend filter.
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
    
    # Get 1d data for HTF trend and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for HTF trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate choppiness regime on 1d (CHOP > 61.8 = range, CHOP < 38.2 = trend)
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1d - lowest_low_1d
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_1d = 100 * np.log10(np.sum(atr_1d, axis=0) / chop_denom * np.sqrt(14)) if False else \
              100 * np.log10(np.nansum(pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values / chop_denom) / 14)
    # Correct CHOP calculation: CHOP = 100 * LOG10(SUM(ATR, n) / (MAX(HIGH) - MIN(LOW)) * SQRT(n))
    sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_atr_1d / chop_denom * np.sqrt(14))
    chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)  # fill NaN with neutral
    chop_regime_trending = chop_1d < 38.2  # trending when CHOP < 38.2
    chop_regime_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_trending)
    
    # Calculate Camarilla levels from previous 4h bar
    camarilla_range = (high - low) * 1.1 / 12.0
    camarilla_R1 = close + camarilla_range
    camarilla_S1 = close - camarilla_range
    
    # Shift by 1 to use only completed 4h bar for Camarilla calculation (no look-ahead)
    camarilla_R1 = np.roll(camarilla_R1, 1)
    camarilla_S1 = np.roll(camarilla_S1, 1)
    camarilla_R1[0] = np.nan
    camarilla_S1[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), ATR (14), and Camarilla (1)
    start_idx = max(34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_regime_trending_aligned[i]) or 
            np.isnan(camarilla_R1[i]) or 
            np.isnan(camarilla_S1[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + above 1d EMA34 + 1d volume spike + trending regime
            long_setup = (close[i] > camarilla_R1[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i] and \
                         chop_regime_trending_aligned[i]
            # Short: price breaks below Camarilla S1 + below 1d EMA34 + 1d volume spike + trending regime
            short_setup = (close[i] < camarilla_S1[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i] and \
                          chop_regime_trending_aligned[i]
            
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
            # Exit: price closes below Camarilla S1 OR below 1d EMA34
            if (close[i] < camarilla_S1[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Camarilla R1 OR above 1d EMA34
            if (close[i] > camarilla_R1[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0