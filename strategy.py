#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume confirmation
    # Long when: price breaks above 1d Donchian upper (20) AND price > 1w EMA34 AND volume > 2.0x 20-bar avg
    # Short when: price breaks below 1d Donchian lower (20) AND price < 1w EMA34 AND volume > 2.0x 20-bar avg
    # Exit when: price crosses 1d Donchian midpoint
    # Uses discrete sizing (0.30) targeting 30-100 total trades over 4 years (7-25/year).
    # 1d timeframe reduces trade frequency vs lower TF, minimizing fee drag.
    # 1w EMA34 provides strong trend filter, reducing whipsaw in choppy/ranging markets.
    # High volume threshold (2.0x) ensures only institutional breakouts are taken.
    # Works in bull (breakouts with trend) and bear (only trend-aligned breaks taken).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_window = 20
    donchian_high_1d = pd.Series(high_1d).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_1d = (donchian_high_1d + donchian_low_1d) / 2.0
    
    # Align 1d Donchian levels to 1d timeframe (no shift needed as we're already on 1d)
    donchian_high_aligned = donchian_high_1d
    donchian_low_aligned = donchian_low_1d
    donchian_mid_aligned = donchian_mid_1d
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.30  # 30% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and (close[i] > ema_34_1w_aligned[i]) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (close[i] < ema_34_1w_aligned[i]) and volume_confirmed[i] and position != -1
        
        # Exit conditions
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

name = "1d_1w_donchian_ema34_volume_v2"
timeframe = "1d"
leverage = 1.0