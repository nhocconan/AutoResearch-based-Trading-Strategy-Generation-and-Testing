#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and choppiness regime filter
    # Enter long when price breaks above 20-day high with volume > 1.5x 20-day avg and chop < 61.8 (trending)
    # Enter short when price breaks below 20-day low with volume > 1.5x 20-day avg and chop < 61.8
    # Exit when price crosses the 20-day midpoint (mean reversion within channel)
    # Uses 1w HTF for volume confirmation (more stable than 1d) and 1d for price action
    # Donchian channels provide clear structure, volume confirms participation, chop filter avoids whipsaws
    # Works in bull (continuation breaks) and bear (reversal breaks at extremes)
    # Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for HTF volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(1) over n) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(n) * (HHV - LLV)))
    # Where ATR_sum = sum of true range over n periods
    # We'll use a simplified version: CHOP = 100 * log10(TR_sum / (log10(n) * (HHV - LLV)))
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first period TR
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hhvl = donchian_high - donchian_low  # highest high - lowest low over 20 periods
    chop = 100 * np.log10(tr_sum / (np.log10(14) * hhvl + 1e-10))  # add small epsilon to avoid div by zero
    
    # Align 1w volume to 1d timeframe
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average volume
    avg_volume = pd.Series(volume_1w_aligned).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume_1w_aligned > (1.5 * avg_volume)
    
    # Choppiness regime filter: CHOP < 61.8 indicates trending market (good for breakouts)
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure Donchian bands are valid
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirmed[i]) or np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]  # break above upper band
        breakout_down = close[i] < donchian_low[i]  # break below lower band
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = breakout_up and volume_confirmed[i] and trending_regime[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and trending_regime[i] and position != -1
        
        # Exit conditions: price crosses the midpoint (mean reversion)
        exit_long = (position == 1 and close[i] < donchian_mid[i])
        exit_short = (position == -1 and close[i] > donchian_mid[i])
        
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

name = "1d_1w_donchian_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0