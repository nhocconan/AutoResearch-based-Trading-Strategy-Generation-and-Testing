#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ChopFilter_v1
Hypothesis: Camarilla R3/S3 breakouts on 12h with 1d EMA34 trend filter, volume spike confirmation, and chop regime filter. 
Camarilla pivot levels provide high-probability support/resistance derived from prior day's range. 
Breakouts above R3 or below S3 with volume expansion and trend alignment capture sustained moves. 
Chop filter avoids whipsaws in ranging markets. Targeting 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous day)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use the prior completed 1d bar to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for each 1d bar
    camarilla_r3 = close_1d_vals + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d_vals - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (1d -> 12h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike detection: volume > 2.5 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.5 * avg_volume)
    
    # Choppiness Index filter to avoid ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # We use a simplified version: CHOP > 61.8 = range, CHOP < 38.2 = trend
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate rolling sum of ATR(14) and range
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop = np.zeros_like(close)
    mask = (range_14 > 0) & (sum_atr_14 > 0)
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Chop filter: only trade when market is trending (CHOP < 45)
    chop_filter = chop < 45
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA34)
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Long logic: price breaks above camarilla R3 with volume spike + in uptrend + trending market
        if (close[i] > camarilla_r3_aligned[i] and 
            volume_spike[i] and 
            uptrend and 
            chop_filter[i]):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below camarilla S3 with volume spike + in downtrend + trending market
        elif (close[i] < camarilla_s3_aligned[i] and 
              volume_spike[i] and 
              downtrend and 
              chop_filter[i]):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite camarilla level or trend weakens or market becomes choppy
        elif position == 1 and (close[i] < camarilla_s3_aligned[i] or not uptrend or not chop_filter[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > camarilla_r3_aligned[i] or not downtrend or not chop_filter[i]):
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

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0