#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA21_VolumeRegime
Hypothesis: Camarilla R1/S1 breakouts on 4h with 12h EMA21 trend filter and volume regime filter. Only trades when volume is above 50th percentile of 100-bar volume (avoids low-volume false breakouts). Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull/bear markets via trend alignment and volume regime filter.
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
    
    # Get 12h data for HTF trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 22:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA21 on 12h close for trend filter
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate Camarilla levels on 12h data (based on previous bar's OHLC)
    camarilla_r1_12h = close_12h + ((high_12h - low_12h) * 1.1 / 12)
    camarilla_s1_12h = close_12h - ((high_12h - low_12h) * 1.1 / 12)
    camarilla_h4_12h = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    camarilla_l4_12h = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align HTF indicators to 4h timeframe
    ema21_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4_12h, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4_12h, additional_delay_bars=1)
    
    # Volume regime: volume above 50th percentile of 100-bar volume (avoids low-volume noise)
    volume_percentile = pd.Series(volume).rolling(window=100, min_periods=100).quantile(0.5).values
    volume_regime = volume > volume_percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA21 (21) and volume percentile (100)
    start_idx = max(21, 100)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_percentile[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with trend filter and volume regime
            # Long: price breaks above R1 in uptrend (close > EMA21) with volume regime
            # Short: price breaks below S1 in downtrend (close < EMA21) with volume regime
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema21_aligned[i]) and volume_regime[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema21_aligned[i]) and volume_regime[i]
            
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
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA21_VolumeRegime"
timeframe = "4h"
leverage = 1.0