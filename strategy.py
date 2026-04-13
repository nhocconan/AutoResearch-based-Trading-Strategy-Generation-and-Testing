#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + chop regime filter
    # Long when: price breaks above Camarilla H3 (1d) AND volume > 2.0x 20-bar avg AND chop > 61.8 (range)
    # Short when: price breaks below Camarilla L3 (1d) AND volume > 2.0x 20-bar avg AND chop > 61.8 (range)
    # Exit when: price returns to Camarilla Pivot (1d) OR chop < 38.2 (trend)
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Camarilla levels provide institutional support/resistance; volume confirms breakout validity;
    # chop filter ensures we only trade in ranging markets where mean reversion works.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # L4 = close - 1.5*(high-low), L3 = close - 1.0*(high-low)
    # We'll use H3/L3 for breakouts and Pivot for exit
    rng = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * rng
    camarilla_l3 = close_1d - 1.0 * rng
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0  # Classic pivot
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 12h chopiness index (Ehler's Chop) for regime filter
    # CHOP = 100 * log10(sum(ATR1) / (n * (max(high)-min(low)))) / log10(n)
    # We'll use a simplified version: high-low range over n periods vs sum of true ranges
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first period
    
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (14 * (max_high - min_low))) / np.log10(14)
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(chop[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # break above previous H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # break below previous L3
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        in_range = chop[i] > 61.8
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = breakout_up and volume_confirmed[i] and in_range and position != 1
        short_entry = breakout_down and volume_confirmed[i] and in_range and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < camarilla_pivot_aligned[i] or chop[i] < 38.2))
        exit_short = (position == -1 and (close[i] > camarilla_pivot_aligned[i] or chop[i] < 38.2))
        
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

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0