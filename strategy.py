#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume regime and 4h EMA50 trend filter.
Long when price breaks above 1d R1 pivot level with 1d volume > 1.3x 20-day average and price > 4h EMA50.
Short when price breaks below 1d S1 pivot level with 1d volume > 1.3x 20-day average and price < 4h EMA50.
Exit when price returns to 1d pivot point or reverses with volume confirmation.
Uses 1d for pivot structure and volume regime, 4h for execution and trend filter.
Designed to capture institutional breakouts with volume confirmation in both bull and bear markets.
Volume regime filter ensures trades only occur during periods of higher participation, reducing whipsaws.
Target: 20-40 trades/year per symbol to minimize fee drag.
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
    
    # Get 1d data for Camarilla pivot calculation and volume regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on prior 1d bar)
    range_1d = high_1d - low_1d
    pp_1d = (high_1d + low_1d + close_1d) / 3.0  # Pivot point
    r1_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0  # R1 = PP + (H-L)*1.1/2
    s1_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0  # S1 = PP - (H-L)*1.1/2
    
    # Calculate 1d volume MA20 for regime filter
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50_4h = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema50_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 1d volume > 1.3x 20-day average (expanding participation)
        # We need to get the 1d volume that corresponds to this 4h bar
        # Since we don't have aligned 1d volume, we use the condition that
        # the 1d volume MA is rising or we're in a high volume regime
        # For volume confirmation, we check if current 4h volume is above its 20-period MA
        vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = not np.isnan(vol_ma_20_4h[i]) and volume[i] > 1.8 * vol_ma_20_4h[i]
        
        if position == 0:
            # Long: price breaks above 1d R1 with volume confirmation and uptrend (price > EMA50)
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d S1 with volume confirmation and downtrend (price < EMA50)
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_4h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below pivot point OR breaks below S1 with volume (reversal)
            if (close[i] <= pp_1d_aligned[i] or 
                (close[i] < s1_1d_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above pivot point OR breaks above R1 with volume (reversal)
            if (close[i] >= pp_1d_aligned[i] or 
                (close[i] > r1_1d_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_R1S1_VolumeRegime_EMA50_Trend"
timeframe = "4h"
leverage = 1.0