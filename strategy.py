#!/usr/bin/env python3
# 1d_Camarilla_R2_S2_Breakout_1wTrend_Volume
# Hypothesis: Breakout above/below weekly Camarilla R2/S2 levels with volume >1.8x 30-bar average and trend filter from 1w EMA200.
# Uses weekly Camarilla pivot levels as strong support/resistance. In uptrend (price > EMA200), buy breakout above R2; in downtrend (price < EMA200), sell breakdown below S2.
# Volume filter ensures only high-conviction moves trigger entries. Designed for 15-25 trades/year on 1d timeframe.
# R2/S2 levels are more robust than R1/S1 and filter out minor breakouts, reducing trade frequency.

name = "1d_Camarilla_R2_S2_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get 1w data for EMA trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(200) for trend filter
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * 2 + ema_200_1w[i-1] * 198) / 200
    
    # Calculate weekly Camarilla levels
    # Pivot = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w) / 3
    range_hl = high_1w - low_1w
    # R2 = C + (H-L) * 1.160
    # S2 = C - (H-L) * 1.160
    camarilla_R2 = close_1w + range_hl * 1.160
    camarilla_S2 = close_1w - range_hl * 1.160
    
    # Align 1w EMA200 and Camarilla levels to 1d timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    camarilla_R2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R2)
    camarilla_S2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S2)
    
    # Volume filter: 1d volume / 30-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 30:
        vol_ma[29] = np.mean(volume[0:30])
        for i in range(30, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 29 + volume[i]) / 30
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(camarilla_R2_aligned[i]) or \
           np.isnan(camarilla_S2_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above weekly Camarilla R2 AND volume confirmation AND bullish trend (price > EMA200)
            if close[i] > camarilla_R2_aligned[i] and volume_ratio[i] > 1.8 and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below weekly Camarilla S2 AND volume confirmation AND bearish trend (price < EMA200)
            elif close[i] < camarilla_S2_aligned[i] and volume_ratio[i] > 1.8 and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below weekly Camarilla S2 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S2_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above weekly Camarilla R2 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R2_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals