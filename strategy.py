#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation (>1.8x 24-bar avg volume).
# Uses wider Camarilla levels (R4/S4) for stronger breakout signals that avoid false breakouts in choppy markets.
# EMA50 on 1d ensures alignment with the dominant daily trend, reducing counter-trend trades.
# Volume confirmation filters low-momentum breakouts. Designed for low trade frequency (<100 total 6h trades)
# to minimize fee drag while capturing strong momentum moves in both bull and bear markets.

name = "6h_Camarilla_R4S4_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # R4 = close + 1.1*(high-low)*1.5
    # S4 = close - 1.1*(high-low)*1.5
    prior_1d_high = df_1d['high'].values
    prior_1d_low = df_1d['low'].values
    prior_1d_close = df_1d['close'].values
    
    camarilla_r4 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.5
    camarilla_s4 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.5
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate average volume for confirmation (24-period on 6h = 4 days)
    lookback_vol = 24
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R4, close > 1d EMA50, volume spike (>1.8x avg)
            if (high[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25  # Reduced size to minimize fee drag
                position = 1
            # SHORT: Price breaks below Camarilla S4, close < 1d EMA50, volume spike (>1.8x avg)
            elif (low[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25  # Reduced size to minimize fee drag
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla R4 or volume drops
            if (low[i] < camarilla_r4_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla S4 or volume drops
            if (high[i] > camarilla_s4_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals