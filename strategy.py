#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA200) and volume confirmation (>1.5x 20-bar avg)
    # Enter long on breakout above Donchian high, short on breakout below Donchian low
    # Exit when price crosses Donchian midpoint
    # Weekly trend filter ensures we only trade in the direction of the higher timeframe trend
    # Volume confirmation avoids false breakouts during low participation
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_window = 20
    donchian_high_6h = pd.Series(high_6h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_6h = pd.Series(low_6h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_6h = (donchian_high_6h + donchian_low_6h) / 2.0
    
    # Align 6h Donchian levels to 6h timeframe (no-op but for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high_6h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low_6h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid_6h)
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Trend filter: price above weekly EMA200 = bullish trend (long only)
    # price below weekly EMA200 = bearish trend (short only)
    bullish_trend = close > ema_200_1w_aligned
    bearish_trend = close < ema_200_1w_aligned
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and bullish_trend[i] and volume_confirmed[i] and position != 1
        short_entry = breakout_down and bearish_trend[i] and volume_confirmed[i] and position != -1
        
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

name = "6h_1w_donchian_ema200_trend_volume_v1"
timeframe = "6h"
leverage = 1.0