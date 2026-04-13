#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1w volume confirmation + chop regime filter
    # Enter long when Alligator jaws (13) < teeth (8) < lips (5) with volume > 1.5x 20-bar avg and CHOP > 61.8 (ranging)
    # Enter short when Alligator jaws > teeth > lips with volume confirmation and CHOP > 61.8
    # Exit when Alligator lines crossover (jaws crosses teeth) or CHOP < 38.2 (strong trend)
    # Uses 1w HTF for volume average (more stable) and 1d for CHOP calculation
    # Williams Alligator identifies trend initiation/continuation
    # Volume confirmation ensures breakouts have participation
    # Chop filter avoids whipsaws in strong trends, favors ranging markets where Alligator excels
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1w data for volume average (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Get 1d data for CHOP calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 12h: SMAs of median price
    # Median price = (high + low) / 2
    median_price_12h = (high_12h + low_12h) / 2.0
    
    # Alligator lines: jaws (13), teeth (8), lips (5)
    # Smoothed with 3-period offset as per Williams
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe (already aligned via get_htf_data)
    # No additional alignment needed as we're using 12h data directly
    
    # Volume confirmation: volume > 1.5x 20-bar average volume (using 1w volume for stability)
    avg_volume_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    volume_confirmed = volume > (1.5 * avg_volume_1w_aligned)
    
    # Choppiness Index on 1d: CHOP > 61.8 = ranging (good for Alligator), CHOP < 38.2 = trending (avoid)
    # True Range = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) = sum of TR over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(sum_tr14 / (atr14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    
    # Align CHOP to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_confirmed[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        # Long: jaws < teeth < lips (alligator sleeping, waking up bullish)
        alligator_long = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        # Short: jaws > teeth > lips (alligator sleeping, waking up bearish)
        alligator_short = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Entry conditions with volume confirmation and chop filter (ranging market)
        long_entry = alligator_long and volume_confirmed[i] and (chop_aligned[i] > 61.8) and position != 1
        short_entry = alligator_short and volume_confirmed[i] and (chop_aligned[i] > 61.8) and position != -1
        
        # Exit conditions
        # Exit on Alligator crossover (jaws crosses teeth) or strong trend (chop < 38.2)
        exit_long = (position == 1) and ((jaw[i] > teeth[i]) or (chop_aligned[i] < 38.2))
        exit_short = (position == -1) and ((jaw[i] < teeth[i]) or (chop_aligned[i] < 38.2))
        
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

name = "12h_1w_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0