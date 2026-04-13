#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
    # Long when: price breaks above 6h Camarilla R4 AND 1w close > 1w EMA(34) AND volume > 1.5x 20-bar avg
    # Short when: price breaks below 6h Camarilla S4 AND 1w close < 1w EMA(34) AND volume > 1.5x 20-bar avg
    # Exit when: price crosses 6h Camarilla pivot point (PP)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years (12-37/year).
    # Weekly EMA filter ensures we only take breakouts in the direction of the higher timeframe trend.
    # Works in bull (breakouts with weekly uptrend) and bear (breakouts with weekly downtrend).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla pivots (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate previous 6h Camarilla levels
    # Camarilla uses previous period's high, low, close
    prev_high_6h = np.roll(high_6h, 1)
    prev_low_6h = np.roll(low_6h, 1)
    prev_close_6h = np.roll(close_6h, 1)
    prev_high_6h[0] = np.nan
    prev_low_6h[0] = np.nan
    prev_close_6h[0] = np.nan
    
    # Camarilla pivot point
    pp_6h = (prev_high_6h + prev_low_6h + prev_close_6h) / 3.0
    # Camarilla levels
    r4_6h = pp_6h + ((prev_high_6h - prev_low_6h) * 1.1 / 2)
    r3_6h = pp_6h + ((prev_high_6h - prev_low_6h) * 1.1 / 4)
    s3_6h = pp_6h - ((prev_high_6h - prev_low_6h) * 1.1 / 4)
    s4_6h = pp_6h - ((prev_high_6h - prev_low_6h) * 1.1 / 2)
    
    # Align 6h Camarilla levels to 6h timeframe (no shift needed as we use previous bar's levels)
    pp_6h_aligned = align_htf_to_ltf(prices, df_6h, pp_6h)
    r4_6h_aligned = align_htf_to_ltf(prices, df_6h, r4_6h)
    r3_6h_aligned = align_htf_to_ltf(prices, df_6h, r3_6h)
    s3_6h_aligned = align_htf_to_ltf(prices, df_6h, s3_6h)
    s4_6h_aligned = align_htf_to_ltf(prices, df_6h, s4_6h)
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly trend filter: close > EMA(34) for long, close < EMA(34) for short
    weekly_uptrend = close_1w > ema_34_1w
    weekly_downtrend = close_1w < ema_34_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(5, n):
        # Skip if data not ready
        if (np.isnan(pp_6h_aligned[i]) or np.isnan(r4_6h_aligned[i]) or np.isnan(s4_6h_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > r4_6h_aligned[i]  # break above R4
        breakout_down = close[i] < s4_6h_aligned[i]  # break below S4
        
        # Entry conditions with weekly trend filter and volume confirmation
        long_entry = breakout_up and weekly_uptrend_aligned[i] and volume_confirmed[i] and position != 1
        short_entry = breakout_down and weekly_downtrend_aligned[i] and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < pp_6h_aligned[i])
        exit_short = (position == -1 and close[i] > pp_6h_aligned[i])
        
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

name = "6h_1w_camarilla_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0