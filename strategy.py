#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeRegime_v1
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1-week EMA50 trend filter, volume confirmation (>1.8x 20-bar MA), and choppiness regime filter (CHOP<40) produce high-quality trades. The 1w trend filter captures major bull/bear regimes while avoiding whipsaws. Chop filter ensures breakouts occur only in strong trending conditions. Target: 100-180 total trades over 4 years (25-45/year) with discrete sizing 0.25 to minimize fee churn.
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
    
    # Load 1w data ONCE before loop for HTF trend filter (EMA50) and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w Camarilla levels (R1, S1) using previous 1w's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shifted = np.concatenate([[np.nan], close_1w[:-1]])  # previous 1w close
    
    camarilla_range = high_1w - low_1w
    r1 = close_1w_shifted + 1.1 * camarilla_range / 12
    s1 = close_1w_shifted - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # 4h volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h choppiness index: CHOP < 40 = strong trending (favor breakouts)
    tr1 = np.maximum(high - low, np.absolute(high - np.concatenate([[np.nan], close[:-1]])))
    tr2 = np.maximum(tr1, np.absolute(low - np.concatenate([[np.nan], close[:-1]])))
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid div by zero
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA, 14*2 for chop)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Choppiness regime: only take breakouts when CHOP < 40 (strong trending)
        regime_ok = chop[i] < 40.0
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume and strong regime
        if uptrend and volume_spike and breakout_r1 and regime_ok:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S1 in downtrend with volume and strong regime
        elif downtrend and volume_spike and breakout_s1 and regime_ok:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend OR regime becomes too choppy (CHOP >= 61.8)
        elif position == 1 and (not uptrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or chop[i] >= 61.8):
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

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0