#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and 1w chop regime filter
# - Long when Williams %R(14) < -80 (oversold) + volume > 1.5x 20-period EMA + CHOP(14) > 61.8 (ranging market)
# - Short when Williams %R(14) > -20 (overbought) + volume > 1.5x 20-period EMA + CHOP(14) > 61.8 (ranging market)
# - Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
# - Position sizing: 0.25 discrete level
# - Targets ~20-30 trades/year on 4h timeframe. Williams %R captures extreme reversals,
#   volume confirmation validates reversal strength, chop filter ensures mean-reversion environment.
#   Works in bull/bear: mean reversion works in ranging markets, chop filter avoids trending markets.

name = "4h_1d_1w_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) from 1d
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high_1d - close_1d) / np.where((highest_high_1d - lowest_low_1d) == 0, 1e-10, (highest_high_1d - lowest_low_1d))
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d volume EMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_ema_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ema_20_1d)
    
    # Calculate 1w Choppiness Index(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CI = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_14 - ll_14
    choppiness = 100 * np.log10(tr_sum_14 / np.where(hh_ll_diff == 0, 1e-10, hh_ll_diff)) / np.log10(14)
    choppiness_aligned = align_htf_to_ltf(prices, df_1w, choppiness)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_ema_20_1d_aligned[i]) or 
            np.isnan(choppiness_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_ema_20_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (mean revert)
        regime_filter = choppiness_aligned[i] > 61.8
        
        # Williams %R entry conditions
        # Long: oversold (%R < -80) + volume confirmation + ranging market
        # Short: overbought (%R > -20) + volume confirmation + ranging market
        long_entry = (williams_r_aligned[i] < -80 and 
                     vol_confirm and 
                     regime_filter)
        short_entry = (williams_r_aligned[i] > -20 and 
                      vol_confirm and 
                      regime_filter)
        
        # Williams %R exit conditions
        # Exit long when %R crosses above -50
        # Exit short when %R crosses below -50
        exit_long = williams_r_aligned[i] > -50
        exit_short = williams_r_aligned[i] < -50
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals