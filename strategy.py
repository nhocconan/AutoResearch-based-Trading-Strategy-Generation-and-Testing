#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeRegime
Hypothesis: 12h Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above R1 in 1w uptrend (close > 1w EMA50) with volume > 2x 20-bar avg.
Short when price breaks below S1 in 1w downtrend (close < 1w EMA50) with volume > 2x 20-bar avg.
Exit when price re-enters H3-L3 range or volume drops below average.
Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull/bear markets via 1w trend filter.
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
    
    # Get 1w data for Camarilla pivot calculation and trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for previous 1w bar
    prev_close = np.concatenate([[np.nan], close_1w[:-1]])
    prev_high = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low = np.concatenate([[np.nan], low_1w[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    h3 = prev_close + range_hl * 1.1 / 4
    l3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Get 1w data for trend filter (EMA50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            if close[i] > ema_trend:  # 1w Uptrend regime
                # Long: break above R1 with volume regime
                long_signal = (close[i] > r1_aligned[i]) and vol_regime[i]
                # Short: break below S1 only if extreme volume (counter-trend fade on very high volume)
                short_signal = (close[i] < s1_aligned[i]) and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # 1w Downtrend regime
                # Short: break below S1 with volume regime
                short_signal = (close[i] < s1_aligned[i]) and vol_regime[i]
                # Long: break above R1 only if extreme volume (counter-trend fade on very high volume)
                long_signal = (close[i] > r1_aligned[i]) and (volume[i] > (4.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: re-enter H3-L3 range or volume drops below average
            exit_signal = (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or (volume[i] < vol_ma_20[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter H3-L3 range or volume drops below average
            exit_signal = (close[i] > l3_aligned[i] and close[i] < h3_aligned[i]) or (volume[i] < vol_ma_20[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0