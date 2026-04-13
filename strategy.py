#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR filter
    # Enter long when price breaks above 4h Donchian upper (20-period) with 1d volume > 1.5x 20-bar avg
    # Enter short when price breaks below 4h Donchian lower (20-period) with 1d volume > 1.5x 20-bar avg
    # Exit when price crosses the 4h Donchian midpoint (10-period average of upper/lower)
    # Uses 1d HTF for volume confirmation (more stable than 4h) and 4h for price channels
    # Donchian channels provide clear structure; volume confirms participation
    # ATR filter avoids entries during low volatility (chop)
    # Works in bull (continuation breaks) and bear (reversal breaks at extremes)
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary timeframe (price channels)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper = highest high over 20 periods, Lower = lowest low over 20 periods
    high_roll_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align 4h Donchian levels to 15m timeframe (if needed) - but we're using 4h as primary
    # Since timeframe is 4h, we need to align to 4h bars directly
    # For 4h timeframe, prices are already at 4h resolution, so no alignment needed
    # However, to be safe and follow MTF rules, we'll use the 4h data directly
    
    # Calculate 1d average volume for confirmation
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume_1d > (1.5 * avg_volume_1d)
    
    # Calculate ATR for volatility filter (14-period ATR on 4h)
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_filter = atr > 0  # avoid zero ATR; in practice, ATR > 0.001 * price
    
    # Align 1d volume confirmation and ATR to 4h timeframe
    volume_confirmed_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    atr_filter_aligned = atr_aligned > (0.001 * close_4h)  # ATR > 0.1% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure Donchian is valid
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirmed_aligned[i]) or np.isnan(atr_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]  # break above upper channel
        breakout_down = close[i] < donchian_lower[i]  # break below lower channel
        
        # Entry conditions with volume confirmation and ATR filter
        long_entry = breakout_up and volume_confirmed_aligned[i] and atr_filter_aligned[i] and position != 1
        short_entry = breakout_down and volume_confirmed_aligned[i] and atr_filter_aligned[i] and position != -1
        
        # Exit conditions: cross the midpoint
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

name = "4h_1d_donchian_breakout_volume_filter_v1"
timeframe = "4h"
leverage = 1.0