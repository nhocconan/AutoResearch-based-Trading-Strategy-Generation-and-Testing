#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_donchian_breakout_volume_trend_v1
# Breakout from 1d Donchian channel with volume confirmation and 12h EMA trend filter.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Low trade frequency expected due to 12h timeframe and multiple confirmation requirements.
name = "12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 12h EMA21 for trend filter
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if indicators not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average 1d volume
        vol_confirm = volume[i] > 1.5 * avg_vol_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]  # price above upper Donchian band
        breakout_down = close[i] < donchian_low_aligned[i]  # price below lower Donchian band
        
        # Trend filter: EMA21 direction
        uptrend = close[i] > ema_21[i]
        downtrend = close[i] < ema_21[i]
        
        # Entry signals: breakout in direction of trend with volume confirmation
        long_entry = breakout_up and uptrend and vol_confirm
        short_entry = breakout_down and downtrend and vol_confirm
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = breakout_down or (close[i] < ema_21[i] and position == 1)
        exit_short = breakout_up or (close[i] > ema_21[i] and position == -1)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals