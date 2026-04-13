#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h strategy using 1w Williams %R for extreme overbought/oversold,
    # confirmed by 1d volume spike and choppiness regime filter.
    # Williams %R identifies reversals in ranging markets; volume confirms momentum;
    # chop filter ensures we avoid strong trends where mean reversion fails.
    # Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
    # Works in both bull and bear: %R captures reversals, volume confirms, chop filter adapts to regime.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for Williams %R (overbought/oversold)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (period=14)
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w + 1e-10)
    
    # Get 1d data for volume confirmation and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d choppiness index (period=14)
    tr_1d = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])), np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_sum_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10((max_high_1d - min_low_1d) / (atr_sum_1d + 1e-10)) * np.log10(14)
    chop_1d = 100 * (np.log10(atr_sum_1d + 1e-10) / np.log10(14)) / (chop_denom + 1e-10)
    
    # Align all HTF indicators to 12h primary timeframe
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_1w_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get 1d bar index for current 12h bar (each 1d bar = 2 12h bars)
        idx_1d = i // 2
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_confirmed = volume_1d[idx_1d] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Chop regime filter: only trade when choppy (CHOP > 61.8 = ranging market)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        # Williams %R extreme levels: oversold < -80, overbought > -20
        williams_r = williams_r_1w_aligned[i]
        oversold = williams_r < -80
        overbought = williams_r > -20
        
        # Entry conditions: mean reversion in ranging markets with volume confirmation
        enter_long = oversold and volume_confirmed and chop_regime
        enter_short = overbought and volume_confirmed and chop_regime
        
        # Exit conditions: reverse signal or chop regime ends
        exit_long = position == 1 and (overbought or not chop_regime)
        exit_short = position == -1 and (oversold or not chop_regime)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0