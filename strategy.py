#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d HMA34 trend filter and volume spike confirmation (>2.0x 20-bar avg volume).
# Uses 4h timeframe for optimal trade frequency (target: 20-50 trades/year). Camarilla R3/S3 levels from prior 1d provide structured breakout zones.
# 1d HMA34 offers smooth trend alignment to avoid counter-trend trades. Volume spike >2.0x average confirms institutional participation.
# Exits on reverse Camarilla level (S3 for longs, R3 for shorts) or volume drop below 50% of average.
# Designed for low trade frequency (<400 total 4h trades) to minimize fee drag while capturing strong momentum moves in both bull and bear markets via trend-following breakouts.

name = "4h_Camarilla_R3S3_Breakout_1dHMA34_VolumeSpike_v1"
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
    
    # Calculate 1d HMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half = 34 // 2
    sqrt_n = int(np.sqrt(34))
    wma_half = pd.Series(close_1d).ewm(span=half, adjust=False).mean().values
    wma_full = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_34_1d = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
    hma_34_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_34_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # R3 = close + 1.1*(high-low)*1.25/4
    # S3 = close - 1.1*(high-low)*1.25/4
    prior_1d_high = df_1d['high'].values
    prior_1d_low = df_1d['low'].values
    prior_1d_close = df_1d['close'].values
    
    camarilla_r3 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.25 / 4
    camarilla_s3 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.25 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(hma_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d HMA34, volume spike (>2.0x avg)
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > hma_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d HMA34, volume spike (>2.0x avg)
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < hma_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla S3 or volume drops below 50% of average
            if (low[i] < camarilla_s3_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla R3 or volume drops below 50% of average
            if (high[i] > camarilla_r3_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals