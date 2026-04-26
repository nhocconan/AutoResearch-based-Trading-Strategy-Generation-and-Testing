#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume_v2
Hypothesis: Trade daily Camarilla R1/S1 breakouts with weekly trend filter and volume confirmation.
Camarilla levels derived from prior week's OHLC act as strong support/resistance. Breakouts with
above-average volume and aligned weekly EMA50 trend capture sustained moves. Designed for very low
trade frequency (7-25/year) on 1d timeframe to minimize fee drag. Uses discrete position sizing (0.25)
and works in bull/bear markets by following weekly EMA50 trend. This addresses the neg_sharpe failure
of #90484 by using weekly (not 1d) OHLC for Camarilla calculation and adding volume confirmation.
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
    
    # Get weekly data for HTF trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous week's OHLC
    # Need to shift weekly data by 1 to get prior week's values
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    
    # Camarilla R1, R2, S1, S2 from prior week
    camarilla_r1 = prev_close_1w + 1.125 * (prev_high_1w - prev_low_1w)
    camarilla_r2 = prev_close_1w + 1.250 * (prev_high_1w - prev_low_1w)
    camarilla_s1 = prev_close_1w - 1.125 * (prev_high_1w - prev_low_1w)
    camarilla_s2 = prev_close_1w - 1.250 * (prev_high_1w - prev_low_1w)
    
    # Align HTF indicators to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s2)
    
    # Volume confirmation: 1.5x median volume (50-period)
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly EMA (50), volume median (50)
    start_idx = max(50, 50)  # EMA50 on 1w, vol median 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_r2_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_s2_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long: break above R1 with volume and weekly uptrend (close > weekly EMA50)
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 1.5 * vol_median_val) and \
                          (close_val > ema_50_1w_val)
            
            # Short: break below S1 with volume and weekly downtrend (close < weekly EMA50)
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 1.5 * vol_median_val) and \
                           (close_val < ema_50_1w_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 (reversal) or trend changes (close < weekly EMA50)
            if (close_val < camarilla_s1_aligned[i]) or \
               (close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 (reversal) or trend changes (close > weekly EMA50)
            if (close_val > camarilla_r1_aligned[i]) or \
               (close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume_v2"
timeframe = "1d"
leverage = 1.0