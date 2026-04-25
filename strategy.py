#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_ChopRegime
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts aligned with weekly trend direction and filtered by daily choppiness regime capture significant moves with minimal trades. Weekly trend ensures alignment with major market direction, while chop regime avoids false breakouts in ranging markets. Designed for 12-30 trades/year to stay within fee-efficient range.
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
    
    # Get weekly data for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data for Camarilla and chop
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla R1 and S1 from previous daily bar
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily choppiness regime: CHOP > 61.8 = ranging (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
    # Using 14-period chop: CHOP = 100 * log10(sum(ATR14) / (log10(n) * (max(high_n) - min(low_n))))
    # Simplified: use ATR ratio and range expansion
    atr14_1d = pd.Series(abs(high_1d - low_1d)).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    sum_atr14 = pd.Series(atr14_1d).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(sum_atr14 / (np.log10(14) * range_14 + 1e-10))
    # Avoid division by zero and invalid values
    chop_raw = np.where(range_14 > 0, chop_raw, 50.0)
    chop_raw = np.where(np.isnan(chop_raw), 50.0, chop_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50 and daily indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only take breakouts when chop < 50 (trending market favors breakouts)
        # In choppy markets (chop > 61.8), avoid breakout entries
        trending_regime = chop_aligned[i] < 50
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-bar average
            volume_confirm = volume[i] > 1.5 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R1 in weekly uptrend with volume and regime
            # Short: price breaks below Camarilla S1 in weekly downtrend with volume and regime
            long_signal = (close[i] > camarilla_r1_aligned[i]) and \
                         (close[i] > ema50_1w_aligned[i]) and \
                         volume_confirm and \
                         trending_regime
            short_signal = (close[i] < camarilla_s1_aligned[i]) and \
                          (close[i] < ema50_1w_aligned[i]) and \
                          volume_confirm and \
                          trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below weekly EMA50 (trend reversal)
            exit_signal = close[i] < ema50_1w_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above weekly EMA50 (trend reversal)
            exit_signal = close[i] > ema50_1w_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_ChopRegime"
timeframe = "12h"
leverage = 1.0