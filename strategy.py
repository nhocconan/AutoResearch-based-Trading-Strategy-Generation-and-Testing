#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when close breaks above Donchian upper band (20-period high) AND price > 1d EMA50 AND volume > 2.0 * 12h volume MA(20);
         Short when close breaks below Donchian lower band (20-period low) AND price < 1d EMA50 AND volume > 2.0 * 12h volume MA(20).
- Exit: Long exits when close crosses below Donchian lower band; Short exits when close crosses above Donchian upper band.
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian channels for clear structure, volume confirmation for participation, and EMA50 trend filter to avoid counter-trend trades.
  Designed to work in both bull (trend following) and bear (mean reversion via exits) markets.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 for 1d trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Upper band = 20-period high, Lower band = 20-period low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian/vol MA need 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
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
                # Long: Close breaks above Donchian upper band AND price > 1d EMA50 (uptrend)
                if curr_close > donchian_high[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below Donchian lower band AND price < 1d EMA50 (downtrend)
                elif curr_close < donchian_low[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below Donchian lower band
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above Donchian upper band
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0