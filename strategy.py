#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ADX trend filter
    # Enter long when price breaks above 20-bar high with volume > 1.5x 20-bar avg and ADX > 25
    # Enter short when price breaks below 20-bar low with volume > 1.5x 20-bar avg and ADX > 25
    # Exit when price crosses the 20-bar midpoint (mean reversion)
    # Uses 1d HTF for volume and ADX (more stable) and 12h for price action
    # Volume confirmation ensures breakouts have participation
    # ADX filter ensures we only trade in trending markets, reducing whipsaws
    # Works in bull (continuation breaks) and bear (trend continuation breaks)
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
    
    # Get 1d data for HTF indicators (volume and ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 12h
    donchian_period = 20
    upper_channel = pd.Series(high_12h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_12h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > (1.5 * avg_volume_1d)
    
    # Calculate 1d ADX for trend filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        if period < len(high):
            plus_sm = np.mean(plus_dm[1:period+1])
            minus_sm = np.mean(minus_dm[1:period+1])
            plus_di[period] = (plus_sm / atr[period]) * 100 if atr[period] != 0 else 0
            minus_di[period] = (minus_sm / atr[period]) * 100 if atr[period] != 0 else 0
            dx[period] = (abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period])) * 100 if (plus_di[period] + minus_di[period]) != 0 else 0
            
            for i in range(period+1, len(high)):
                plus_sm = (plus_sm * (period-1) + plus_dm[i]) / period
                minus_sm = (minus_sm * (period-1) + minus_dm[i]) / period
                plus_di[i] = (plus_sm / atr[i]) * 100 if atr[i] != 0 else 0
                minus_di[i] = (minus_sm / atr[i]) * 100 if atr[i] != 0 else 0
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(dx)
        if 2*period < len(dx):
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_filter = adx_1d > 25  # Only trade when ADX > 25 (trending market)
    
    # Align HTF indicators to 12h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    middle_channel_aligned = align_htf_to_ltf(prices, df_12h, middle_channel)
    volume_confirmed_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d)
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or np.isnan(middle_channel_aligned[i]) or
            np.isnan(volume_confirmed_aligned[i]) or np.isnan(adx_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_channel_aligned[i]  # break above upper channel
        breakout_down = close[i] < lower_channel_aligned[i]  # break below lower channel
        
        # Entry conditions with volume confirmation and ADX filter
        long_entry = breakout_up and volume_confirmed_aligned[i] and adx_filter_aligned[i] and position != 1
        short_entry = breakout_down and volume_confirmed_aligned[i] and adx_filter_aligned[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < middle_channel_aligned[i])
        exit_short = (position == -1 and close[i] > middle_channel_aligned[i])
        
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

name = "12h_1d_donchian_volume_adx_filter_v1"
timeframe = "12h"
leverage = 1.0