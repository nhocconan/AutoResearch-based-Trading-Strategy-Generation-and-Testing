#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R with 1-day Trend and Volume Confirmation.
Long when Williams %R < -80 (oversold) and 1-day EMA50 rising with volume spike.
Short when Williams %R > -20 (overbought) and 1-day EMA50 falling with volume spike.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Williams %R identifies reversals in both bull and bear markets, while 1-day trend filter
ensures we trade with the higher timeframe direction. Volume confirmation reduces false signals.
Designed for low trade frequency (<400 total) by requiring multiple confirmations.
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
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), 1d EMA50 rising, volume spike
            if (williams_r[i] < -80 and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), 1d EMA50 falling, volume spike
            elif (williams_r[i] > -20 and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R >= -50
                if williams_r[i] >= -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R <= -50
                if williams_r[i] <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0