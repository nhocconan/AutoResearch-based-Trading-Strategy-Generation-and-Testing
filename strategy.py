#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout + 4h EMA200 trend filter + volume confirmation + session filter (08-20 UTC)
    # Long when: price breaks above Camarilla H3 (1h) AND price > 4h EMA200 AND volume > 1.5x 20-bar avg AND session 08-20 UTC
    # Short when: price breaks below Camarilla L3 (1h) AND price < 4h EMA200 AND volume > 1.5x 20-bar avg AND session 08-20 UTC
    # Exit when: price crosses Camarilla pivot point (PP)
    # Uses discrete sizing (0.20) targeting 60-150 total trades over 4 years.
    # Camarilla provides intraday support/resistance; 4h EMA200 filters trend; volume confirms breakout strength.
    # Works in bull (breakouts with trend) and bear (strong trend-aligned breaks only).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA200
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    # Camarilla levels for intraday trading
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 4
    # L3 = PP - (H - L) * 1.1 / 4
    
    # Need daily data to calculate Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for each 1h bar
    # We'll calculate Camarilla levels using the previous completed day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    PP = (high_1d + low_1d + close_1d) / 3.0
    H3 = PP + (high_1d - low_1d) * 1.1 / 4.0
    L3 = PP - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe (using previous day's levels)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP, additional_delay_bars=1)  # previous day's PP
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3, additional_delay_bars=1)  # previous day's H3
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3, additional_delay_bars=1)  # previous day's L3
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(1, n):  # Start from 1 to access i-1 for breakout
        # Skip if data not ready
        if (np.isnan(PP_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            # Outside session: flatten position
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > H3_aligned[i-1]  # break above previous H3
        breakout_down = close[i] < L3_aligned[i-1]  # break below previous L3
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and (close[i] > ema_200_4h_aligned[i]) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (close[i] < ema_200_4h_aligned[i]) and volume_confirmed[i] and position != -1
        
        # Exit conditions: price crosses pivot point (PP)
        exit_long = (position == 1 and close[i] < PP_aligned[i])
        exit_short = (position == -1 and close[i] > PP_aligned[i])
        
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

name = "1h_camarilla_ema200_volume_session_v1"
timeframe = "1h"
leverage = 1.0