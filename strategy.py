#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d volume and chop regime filter
    # Enter long when price breaks above Donchian(20) high with volume > 1.5x avg and chop < 61.8
    # Enter short when price breaks below Donchian(20) low with volume > 1.5x avg and chop < 61.8
    # Exit when price touches Donchian midpoint (mean reversion in choppy markets)
    # Uses 1d HTF for Donchian calculation (more stable) and 12h for entry/exit timing
    # Donchian channels provide clear structure, volume confirms participation, chop filter avoids whipsaws
    # Works in bull (continuation breaks) and bear (mean reversion at extremes)
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
    
    # Get 1d data for Donchian calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # Choppiness Index regime filter (avoid whipsaws in choppy markets)
    # CHOP = 100 * log10(sum(ATR(14)) / log10((max(high, n) - min(low, n)) * sqrt(n))
    # Simplified: CHOP > 61.8 = choppy (range), CHOP < 38.2 = trending
    # We want trending markets for breakouts: CHOP < 61.8
    tr1 = np.maximum(high[1:], low[:-1]) - np.minimum(low[1:], high[:-1])  # True Range
    tr1 = np.concatenate([[0], tr1])  # align length
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10((max_high - min_low) * np.sqrt(14))
    chop_denom = np.where(chop_denom == 0, np.nan, chop_denom)
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / chop_denom)
    chop = np.where(np.isnan(chop) | (chop_denom == 0), 50, chop)  # default to neutral
    chop_filter = chop < 61.8  # allow breakouts in less choppy (more trending) markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to allow for Donchian calculation
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]  # break above upper band
        breakout_down = close[i] < donchian_low_aligned[i]  # break below lower band
        
        # Entry conditions with volume and chop confirmation
        long_entry = breakout_up and volume_confirmed[i] and chop_filter[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and chop_filter[i] and position != -1
        
        # Exit conditions: return to midpoint (mean reversion)
        exit_long = (position == 1 and close[i] < donchian_mid_aligned[i])
        exit_short = (position == -1 and close[i] > donchian_mid_aligned[i])
        
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

name = "12h_1d_donchian_volume_chop_filter_v1"
timeframe = "12h"
leverage = 1.0