#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1d chop regime filter
# - Long when price breaks above 12h Camarilla R4 level AND 1d volume > 1.5x 20-period volume SMA AND 1d chop > 61.8 (range regime)
# - Short when price breaks below 12h Camarilla S4 level AND 1d volume > 1.5x 20-period volume SMA AND 1d chop > 61.8 (range regime)
# - Exit: price retreats to Camarilla pivot point (PP)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Camarilla levels from 1d timeframe for structure, 12h for execution timing
# - Chop regime filter ensures we only trade in ranging markets where mean reversion works

name = "12h_1d_camarilla_breakout_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla formula: PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    camarilla_r4 = camarilla_pp + camarilla_range * 1.1 / 2.0
    camarilla_s4 = camarilla_pp - camarilla_range * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 12h volume SMA for confirmation
    volume_sma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Chopiness Index for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: Chop > 61.8 = ranging, Chop < 38.2 = trending
    tr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = highest_high_14 - lowest_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Avoid division by zero
    chop_1d = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(volume_sma_20_12h[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA AND 1d volume > 1.5x 20-period volume SMA
        vol_confirm_12h = volume[i] > 1.5 * volume_sma_20_12h[i]
        vol_confirm_1d = volume_1d[i] > 1.5 * volume_sma_20_1d_aligned[i] if i < len(volume_1d) else False
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        # Regime filter: Chop > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_r4_aligned[i-1] if i > 0 else False  # Break above previous R4
        breakout_down = close[i] < camarilla_s4_aligned[i-1] if i > 0 else False  # Break below previous S4
        
        # Exit condition: price retreats to pivot point
        exit_long = close[i] < camarilla_pp_aligned[i]
        exit_short = close[i] > camarilla_pp_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and chop_filter:
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