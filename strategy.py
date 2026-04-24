#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Donchian upper channel AND price > 1d EMA50 AND volume > 1.5 * 12h volume MA(20);
         Short when price breaks below Donchian lower channel AND price < 1d EMA50 AND volume > 1.5 * 12h volume MA(20).
- Exit: Long exits when price breaks below Donchian lower channel; Short exits when price breaks above Donchian upper channel.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with volume confirmation to avoid false breakouts.
- Uses Donchian channels for clear structure with trend alignment to avoid counter-trend trades.
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
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 for 1d
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels from 12h data (20-period)
    high_12h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_12h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(high_12h[i]) or 
            np.isnan(low_12h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above upper Donchian AND price > 1d EMA50 (uptrend)
                if curr_high > high_12h[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian AND price < 1d EMA50 (downtrend)
                elif curr_low < low_12h[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below lower Donchian
            if curr_low < low_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above upper Donchian
            if curr_high > high_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0