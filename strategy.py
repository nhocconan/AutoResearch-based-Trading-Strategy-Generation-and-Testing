#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA34 Trend + Volume Spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when close breaks above Donchian(20) upper band AND price > 1d EMA34 AND volume > 2.0 * 12h volume MA(20);
         Short when close breaks below Donchian(20) lower band AND price < 1d EMA34 AND volume > 2.0 * 12h volume MA(20).
- Exit: Long exits when close crosses below Donchian(20) lower band; Short exits when close crosses above Donchian(20) upper band.
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian breakouts for clear structure, volume confirmation for participation,
  and EMA34 trend filter to avoid counter-trend trades. Proven structure with tight entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 for 1d trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 12h data for Donchian(20) and volume MA(20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for Donchian(20) and volume MA(20)
        return np.zeros(n)
    
    # Calculate Donchian(20) bands for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian bands to 12h timeframe (no alignment needed as already 12h)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    
    # Get 12h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, Donchian/volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above Donchian upper AND price > 1d EMA34 (uptrend)
                if curr_close > donchian_upper_aligned[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below Donchian lower AND price < 1d EMA34 (downtrend)
                elif curr_close < donchian_lower_aligned[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below Donchian lower
            if curr_close < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above Donchian upper
            if curr_close > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0