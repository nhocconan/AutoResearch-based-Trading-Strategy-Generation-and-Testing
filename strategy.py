#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour timeframe with 4-hour Donchian(15) breakout and 1-day EMA(50) trend filter
# Uses higher timeframe (4h/1d) for signal direction to reduce trade frequency
# 1h only for precise entry timing. Volume confirmation ensures institutional participation
# Designed for low frequency (target: 15-30 trades/year) to minimize fee drag in 1h
# Works in both bull and bear markets by aligning with higher timeframe trend
# Session filter (08-20 UTC) reduces noise during low-liquidity periods

name = "1h_donchian15_4h1d_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-market to post-close)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h Donchian channel (15-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=15, min_periods=15).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=15, min_periods=15).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # 1d EMA trend filter (50-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout conditions (using 4h levels)
        breakout_up = close[i] > donchian_high_4h_aligned[i-1] if i > 0 else False
        breakout_down = close[i] < donchian_low_4h_aligned[i-1] if i > 0 else False
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on downside breakout or trend reversal
            if breakout_down or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit on upside breakout or trend reversal
            if breakout_up or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Long on upside breakout in uptrend
            if breakout_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.20
            # Short on downside breakout in downtrend
            elif breakout_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.20
    
    return signals