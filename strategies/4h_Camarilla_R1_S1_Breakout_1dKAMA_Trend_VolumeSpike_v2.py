#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dKAMA_Trend_VolumeSpike_v2
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts aligned with 1d KAMA trend filter and volume confirmation capture strong trends while avoiding whipsaws. Added volume regime filter (low volume chop avoidance) to reduce trades and improve selectivity. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Load 1d data ONCE before loop for HTF trend filter (KAMA) and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for trend filter (adaptive to market noise)
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        net_change = np.abs(close_1d[i] - close_1d[i-10])
        total_change = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
        er[i] = net_change / total_change if total_change != 0 else 0
    # Smoothing constants: fastest SC = 2/(2+1) = 0.666, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.666 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed value
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])  # previous 1d close
    
    camarilla_range = high_1d - low_1d
    r1 = close_1d_shifted + 1.1 * camarilla_range / 12
    s1 = close_1d_shifted - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    # Volume regime: avoid extremely low volume (chop) - volume > 0.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    low_volume_regime = volume < 0.5 * vol_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for KAMA and volume MA)
    start_idx = max(30, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(vol_ma_50[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (KAMA)
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Volume confirmation and regime filter
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        not_low_volume = not low_volume_regime[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume and not in low volume regime
        if uptrend and volume_spike and not_low_volume and breakout_r1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S1 in downtrend with volume and not in low volume regime
        elif downtrend and volume_spike and not_low_volume and breakout_s1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend OR entering low volume regime
        elif position == 1 and (not uptrend or low_volume_regime[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or low_volume_regime[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dKAMA_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0