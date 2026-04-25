#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime
Hypothesis: Trade 4h timeframe using Camarilla pivot levels (R1, S1) from daily candles for breakout entries, 
daily EMA34 for trend filter, daily volume spike (>2.0x 20-bar MA) for confirmation, and choppiness regime filter 
(CHOP > 61.8 = range, CHOP < 38.2 = trend) to avoid whipsaws. Enter long when price breaks above R1 AND above 
daily EMA34 AND volume spike AND trend regime. Enter short when price breaks below S1 AND below daily EMA34 AND 
volume spike AND trend regime. Exit on opposite Camarilla touch or trend reversal. Uses discrete sizing 0.30 to 
balance return and drawdown. Target 75-200 total trades over 4 years (19-50/year). Works in bull/bear via 
Camarilla structure and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, EMA34, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla levels (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1_1d = close_1d + (1.1 * (high_1d - low_1d) / 12)
    camarilla_s1_1d = close_1d - (1.1 * (high_1d - low_1d) / 12)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Calculate choppiness index (CHOP) on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * (max(high) - min(low)))) / log10(n)
    # Using 14-period as standard
    atr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    atr_1d[0] = high_1d[0] - low_1d[0]  # first period
    atr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * (max_high_1d - min_low_1d)
    chop_1d = np.where(chop_denominator != 0, 100 * np.log10(atr_sum_1d / chop_denominator) / np.log10(14), 50)
    chop_regime_1d = (chop_1d > 61.8) | (chop_1d < 38.2)  # True in trending OR ranging (avoid neutral zone 38.2-61.8)
    
    # Align all 1d indicators to 4h timeframe (completed daily bar only)
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (need prev day), EMA34 (34), volume MA (20), CHOP (14)
    start_idx = max(1, 34, 20, 14)  # Camarilla needs previous day's data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_1d_aligned[i]) or np.isnan(camarilla_s1_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(chop_regime_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above daily EMA34 AND volume spike AND chop regime (trend or range)
            long_setup = (close[i] > camarilla_r1_1d_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i] and \
                         chop_regime_1d_aligned[i]
            # Short: price breaks below S1 AND below daily EMA34 AND volume spike AND chop regime
            short_setup = (close[i] < camarilla_s1_1d_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i] and \
                          chop_regime_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price touches S1 OR closes below daily EMA34
            if (close[i] <= camarilla_s1_1d_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price touches R1 OR closes above daily EMA34
            if (close[i] >= camarilla_r1_1d_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0