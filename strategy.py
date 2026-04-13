#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(15) breakout + 1d weekly pivot R4/S4 filter + volume confirmation
    # Long when: price breaks above 12h Donchian upper (15) AND close > weekly R4 AND volume > 1.8x 20-bar avg
    # Short when: price breaks below 12h Donchian lower (15) AND close < weekly S4 AND volume > 1.8x 20-bar avg
    # Exit when: price crosses 12h Donchian midpoint OR weekly pivot PP
    # Uses discrete sizing (0.25) targeting 12-37 trades/year on 12h timeframe.
    # Weekly pivot acts as strong institutional level filter; volume confirms breakout validity.
    # Works in bull (strong continuation breaks) and bear (only strong down breaks traded).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from 5-day rolling window on 1d data
    if len(high_1d) >= 5:
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    else:
        weekly_high = high_1d
        weekly_low = low_1d
        weekly_close = close_1d
    
    PP_weekly = (weekly_high + weekly_low + weekly_close) / 3.0
    R4_weekly = PP_weekly + (weekly_high - weekly_low) * 1.1 / 2.0  # R4 = PP + 1.1*(H-L)
    S4_weekly = PP_weekly - (weekly_high - weekly_low) * 1.1 / 2.0  # S4 = PP - 1.1*(H-L)
    
    # Calculate 12h Donchian channels (15-period for fewer signals)
    donchian_window = 15
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align HTF indicators to 12h timeframe (wait for completed 1d bar)
    R4_weekly_aligned = align_htf_to_ltf(prices, df_1d, R4_weekly)
    S4_weekly_aligned = align_htf_to_ltf(prices, df_1d, S4_weekly)
    PP_weekly_aligned = align_htf_to_ltf(prices, df_1d, PP_weekly)
    
    # Volume confirmation: volume > 1.8x 20-bar average volume (stricter for fewer trades)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(R4_weekly_aligned[i]) or np.isnan(S4_weekly_aligned[i]) or np.isnan(PP_weekly_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # break below previous Donchian low
        
        # Weekly pivot filter: only trade strong breaks beyond R4/S4
        strong_break_up = breakout_up and (close[i] > R4_weekly_aligned[i])
        strong_break_down = breakout_down and (close[i] < S4_weekly_aligned[i])
        
        # Entry conditions with volume confirmation
        long_entry = strong_break_up and volume_confirmed[i] and position != 1
        short_entry = strong_break_down and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < donchian_mid[i] or close[i] < PP_weekly_aligned[i]))
        exit_short = (position == -1 and (close[i] > donchian_mid[i] or close[i] > PP_weekly_aligned[i]))
        
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

name = "12h_1d_donchian_weekly_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0