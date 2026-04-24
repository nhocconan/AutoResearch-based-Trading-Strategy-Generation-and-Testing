#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ATR-based trend filter + volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) and close price for trend filter (ATR expansion indicates trending market).
- Entry: Long when price breaks above Donchian upper(20) AND 1d ATR(14) > 1d ATR(50) AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian lower(20) AND 1d ATR(14) > 1d ATR(50) AND volume > 1.5 * 4h volume MA(20).
- Exit: Long exits when price crosses below Donchian lower(10) for quicker profit taking; Short exits when price crosses above Donchian upper(10).
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian channels provide clear breakout levels; ATR expansion filters for trending regimes; volume spike confirms conviction.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) by requiring volatility expansion.
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
    
    # Get 1d data for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculations
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Calculate Donchian channels on 4h
    donchian_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_lower_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Get 4h data for volume MA(20)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian 20 needs 20, ATR 50 needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_20[i]) or 
            np.isnan(donchian_lower_20[i]) or 
            np.isnan(donchian_upper_10[i]) or 
            np.isnan(donchian_lower_10[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: ATR expansion (shorter ATR > longer ATR indicates trending market)
        trending = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if trending and vol_confirm:
                # Long: price breaks above Donchian upper(20)
                if curr_high > donchian_upper_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian lower(20)
                elif curr_low < donchian_lower_20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Donchian lower(10) for quicker exit
            if curr_low < donchian_lower_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Donchian upper(10)
            if curr_high > donchian_upper_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0