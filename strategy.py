#!/usr/bin/env python3
"""
Hypothesis: 1h 4h Donchian breakout with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 1h for precise entry timing.
- HTF: 4h Donchian(20) for breakout structure, 1d EMA(34) for trend filter.
- Volume: Current 1h volume > 1.5 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above 4h Donchian upper AND price > 1d EMA34 AND volume spike.
         Short when price breaks below 4h Donchian lower AND price < 1d EMA34 AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Only trade 08-20 UTC to avoid low-volume Asian session noise.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Donchian channels provide objective breakout levels; 1d EMA ensures alignment with higher timeframe trend;
volume confirmation reduces false breakouts; session filter improves signal quality.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 1h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current 1h volume > 1.5 * 20-period volume MA
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # Donchian(20), EMA(34), vol MA(20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        ema = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session filter
            if vol_spike:
                # Bullish: price breaks above Donchian upper AND price > 1d EMA34
                if curr_high > dh and curr_close > ema:
                    signals[i] = 0.20
                    position = 1
                # Bearish: price breaks below Donchian lower AND price < 1d EMA34
                elif curr_low < dl and curr_close < ema:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR loss of volume confirmation
            if curr_low < dl or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR loss of volume confirmation
            if curr_high > dh or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_1dEMA34_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0