#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h session-filtered strategy using 4h Camarilla pivot breakout + volume confirmation
    # Long when: price breaks above 4h Camarilla H3 level AND volume > 1.3x 20-bar avg AND 08-20 UTC session
    # Short when: price breaks below 4h Camarilla L3 level AND volume > 1.3x 20-bar avg AND 08-20 UTC session
    # Exit when: price crosses 4h Camarilla pivot point (PP)
    # Uses discrete sizing (0.20) targeting 60-150 total trades over 4 years (15-37/year).
    # Session filter reduces noise trades during low-volume hours.
    # Camarilla levels provide precise intraday support/resistance with statistical edge.
    # Volume confirmation ensures breakouts have institutional participation.
    # Works in bull (buying H3 breaks) and bear (selling L3 breaks) with trend alignment.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (based on previous bar's OHLC)
    camarilla_high_4h = np.roll(high_4h, 1)
    camarilla_low_4h = np.roll(low_4h, 1)
    camarilla_close_4h = np.roll(close_4h, 1)
    
    # Avoid division by zero in first bar
    camarilla_high_4h[0] = camarilla_high_4h[1] if len(camarilla_high_4h) > 1 else camarilla_high_4h[0]
    camarilla_low_4h[0] = camarilla_low_4h[1] if len(camarilla_low_4h) > 1 else camarilla_low_4h[0]
    camarilla_close_4h[0] = camarilla_close_4h[1] if len(camarilla_close_4h) > 1 else camarilla_close_4h[0]
    
    camarilla_range = camarilla_high_4h - camarilla_low_4h
    camarilla_pp = (camarilla_high_4h + camarilla_low_4h + camarilla_close_4h) / 3.0
    camarilla_h3 = camarilla_pp + (camarilla_range * 1.1 / 4.0)
    camarilla_l3 = camarilla_pp - (camarilla_range * 1.1 / 4.0)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    
    # Volume confirmation: volume > 1.3x 20-bar average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (position_size if position == 1 else -position_size)
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous bar's levels)
        breakout_h3 = close[i] > camarilla_h3_aligned[i-1]  # break above previous H3
        breakout_l3 = close[i] < camarilla_l3_aligned[i-1]  # break below previous L3
        
        # Entry conditions with volume confirmation
        long_entry = breakout_h3 and volume_confirmed[i] and position != 1
        short_entry = breakout_l3 and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_pp_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_pp_aligned[i])
        
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

name = "1h_4h_camarilla_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0