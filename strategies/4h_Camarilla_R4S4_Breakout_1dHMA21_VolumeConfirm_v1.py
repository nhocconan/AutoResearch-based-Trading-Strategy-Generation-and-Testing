#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1d HMA21 trend filter and volume confirmation (>1.5x 20-bar avg volume).
# Uses wider Camarilla levels (R4/S4) to capture stronger breakouts, HMA21 for smooth 1d trend alignment,
# and moderate volume threshold to reduce false signals while maintaining sufficient trade frequency.
# Designed for low trade frequency (<150 total 4h trades over 4 years) to minimize fee drag
# while capturing strong momentum moves in both bull and bear markets via trend-following breakouts.

name = "4h_Camarilla_R4S4_Breakout_1dHMA21_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1d HMA21 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).ewm(span=half, adjust=False).mean().values
    wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_1d = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # R4 = close + 1.1*(high-low)*1.5/4
    # S4 = close - 1.1*(high-low)*1.5/4
    prior_1d_high = df_1d['high'].values
    prior_1d_low = df_1d['low'].values
    prior_1d_close = df_1d['close'].values
    
    camarilla_r4 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.5 / 4
    camarilla_s4 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.5 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R4, close > 1d HMA21, volume spike (>1.5x avg)
            if (high[i] > camarilla_r4_aligned[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S4, close < 1d HMA21, volume spike (>1.5x avg)
            elif (low[i] < camarilla_s4_aligned[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla R4 or volume drops
            if (low[i] < camarilla_r4_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla S4 or volume drops
            if (high[i] > camarilla_s4_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals