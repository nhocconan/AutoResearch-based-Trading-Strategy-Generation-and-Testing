#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter, volume spike confirmation, and choppiness regime filter. In trending markets (CHOP < 38.2), trade breakouts in direction of 1d EMA34 trend. In ranging markets (CHOP > 61.8), fade breaks at R1/S1. Uses discrete position sizing (0.25) to minimize fee churn. Targets 50-150 trades over 4 years by requiring confluence of price level, volume, and regime. Works in bull/bear via adaptive logic: momentum in trend, mean reversion in chop.
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
    
    # Load 1d data ONCE before loop for HTF trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for volume normalization (20-period)
    atr_period = 20
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 1d True Range for choppiness indicator
    tr_1d = np.maximum.reduce([
        df_1d['high'].values - df_1d['low'].values,
        np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
        np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    ])
    tr_1d[0] = 0
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate choppiness index (14-period)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla pivot levels from previous 1d
    # R1, S1 levels
    camarilla_r1 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.0833
    camarilla_s1 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.0833
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate volume spike (current volume > 2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for ATR/volume, 14 for chop)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime-based logic using choppiness
        if chop_aligned[i] < 38.2:  # Trending regime
            # Trade breakouts in direction of 1d trend
            if close[i] > camarilla_r1_aligned[i] and volume_spike[i] and htf_trend[i] == 1:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] < camarilla_s1_aligned[i] and volume_spike[i] and htf_trend[i] == -1:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif chop_aligned[i] > 61.8:  # Ranging regime
            # Fade breaks at R1/S1 (mean reversion)
            if close[i] > camarilla_r1_aligned[i] and volume_spike[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            elif close[i] < camarilla_s1_aligned[i] and volume_spike[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Exit fading position when price returns to pivot area
                if position == 1 and close[i] < camarilla_r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > camarilla_s1_aligned[i]:
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
        else:  # Transition regime (38.2 <= CHOP <= 61.8)
            # Hold current position or stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0