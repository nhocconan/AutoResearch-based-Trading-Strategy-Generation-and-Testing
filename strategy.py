#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA(50) for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Donchian upper(20) in uptrend with volume > 1.5 * 12h volume MA(20);
         Short when price breaks below Donchian lower(20) in downtrend with volume > 1.5 * 12h volume MA(20).
- Exit: Opposite Donchian breakout (Long exits when price < Donchian lower(10), Short exits when price > Donchian upper(10)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian channels provide clear structure, EMA50 avoids counter-trend trades, volume confirms conviction.
- Works in bull (breakouts with strength) and bear (breakdowns with volume).
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h Donchian channels (20 for entry, 10 for exit)
    lookback_entry = 20
    lookback_exit = 10
    
    # Initialize arrays for Donchian channels
    dc_upper_entry = np.full(n, np.nan)
    dc_lower_entry = np.full(n, np.nan)
    dc_upper_exit = np.full(n, np.nan)
    dc_lower_exit = np.full(n, np.nan)
    
    # Calculate Donchian channels using rolling window
    for i in range(lookback_entry - 1, n):
        dc_upper_entry[i] = np.max(high[i - lookback_entry + 1:i + 1])
        dc_lower_entry[i] = np.min(low[i - lookback_entry + 1:i + 1])
    
    for i in range(lookback_exit - 1, n):
        dc_upper_exit[i] = np.max(high[i - lookback_exit + 1:i + 1])
        dc_lower_exit[i] = np.min(low[i - lookback_exit + 1:i + 1])
    
    # Calculate 12h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback_entry, 50, 20)  # Donchian needs 20, EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(dc_upper_entry[i]) or 
            np.isnan(dc_lower_entry[i]) or np.isnan(dc_upper_exit[i]) or 
            np.isnan(dc_lower_exit[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * volume_ma[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Donchian upper(20)
                if curr_high > dc_upper_entry[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below Donchian lower(20)
                if curr_low < dc_lower_entry[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below Donchian lower(10)
            if curr_low < dc_lower_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian upper(10)
            if curr_high > dc_upper_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0