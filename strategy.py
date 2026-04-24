#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout + 1d EMA34 Trend Filter + Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when close breaks above Donchian(20) upper band AND price > 1d EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when close breaks below Donchian(20) lower band AND price < 1d EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when close crosses below Donchian(20) middle band (10-period); Short exits when close crosses above middle band.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian breakout captures momentum; EMA34 filters higher-timeframe trend; volume spike confirms conviction.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = highest_high
    lower_band = lowest_low
    middle_band = (upper_band + lower_band) / 2.0
    
    # Get 4h data for volume MA(20)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 34)  # Donchian needs 20, volume MA needs 20, EMA34 needs 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter from 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: close breaks above Donchian upper band
                if curr_close > upper_band[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: close breaks below Donchian lower band
                if curr_close < lower_band[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below middle band
            if curr_close < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above middle band
            if curr_close > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0