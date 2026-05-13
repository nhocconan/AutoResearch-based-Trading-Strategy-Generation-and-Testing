#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses tighter Camarilla levels (R3/S3) to capture stronger breakouts with less noise, 12h EMA50 for smooth trend alignment,
# and higher volume threshold to significantly reduce false signals and trade frequency.
# Target: <100 total 4h trades over 4 years (<25/year) to minimize fee drag while capturing high-probability momentum moves.
# Works in bull/bear markets by only taking breakouts in direction of 12h trend with volume confirmation.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v2"
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels (based on prior 12h bar)
    # R3 = close + 1.1*(high-low)*1.25/4
    # S3 = close - 1.1*(high-low)*1.25/4
    prior_12h_high = df_12h['high'].values
    prior_12h_low = df_12h['low'].values
    prior_12h_close = df_12h['close'].values
    
    camarilla_r3 = prior_12h_close + 1.1 * (prior_12h_high - prior_12h_low) * 1.25 / 4
    camarilla_s3 = prior_12h_close - 1.1 * (prior_12h_high - prior_12h_low) * 1.25 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, price > 12h EMA50, volume spike (>1.8x avg)
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, price < 12h EMA50, volume spike (>1.8x avg)
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla R3 or volume drops
            if (low[i] < camarilla_r3_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla S3 or volume drops
            if (high[i] > camarilla_s3_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals