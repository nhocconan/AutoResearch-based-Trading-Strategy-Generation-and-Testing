#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
    # Long when: price breaks above Camarilla H3 AND 1d volume > 2x 20-bar avg AND chop > 61.8 (trending)
    # Short when: price breaks below Camarilla L3 AND 1d volume > 2x 20-bar avg AND chop > 61.8 (trending)
    # Exit when: price crosses Camarilla pivot point (PP) OR chop < 38.2 (range)
    # Uses discrete sizing (0.25) targeting 75-200 trades over 4 years.
    # Volume spike confirms institutional interest, chop filter ensures trending conditions.
    # Works in bull/bear via directional breakouts and trend filter.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    rng = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_h3 = camarilla_pp + (rng * 1.1 / 4)
    camarilla_l3 = camarilla_pp - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume confirmation: volume > 2x 20-bar average volume
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * avg_vol_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 4h choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    sum_atr = pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = np.where(range_hl > 0, 100 * np.log10(sum_atr / range_hl) / np.log10(atr_period), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Break above H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Break below L3
        
        # Regime filter: trending market (CHOP > 61.8)
        trending_market = chop[i] > 61.8
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = breakout_up and volume_spike_aligned[i] and trending_market and position != 1
        short_entry = breakout_down and volume_spike_aligned[i] and trending_market and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < camarilla_pp_aligned[i] or chop[i] < 38.2))
        exit_short = (position == -1 and (close[i] > camarilla_pp_aligned[i] or chop[i] < 38.2))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
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

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0