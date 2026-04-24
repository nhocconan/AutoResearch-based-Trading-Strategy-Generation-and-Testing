#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5 * 1d volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5 * 1d volume MA(20).
- Exit: Opposite Donchian breakout (Long exits when price < Donchian(10) low, Short exits when price > Donchian(10) high).
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian channels provide structure, EMA50 filters higher-timeframe trend, volume confirms conviction.
- Works in bull (buying breakouts) and bear (selling breakdowns) with reduced whipsaws from 1w trend filter.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data for volume MA(20)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    lookback_entry = 20
    lookback_exit = 10
    
    # Precompute rolling max/min for Donchian channels
    donchian_high_entry = np.full(n, np.nan)
    donchian_low_entry = np.full(n, np.nan)
    donchian_high_exit = np.full(n, np.nan)
    donchian_low_exit = np.full(n, np.nan)
    
    for i in range(lookback_entry, n):
        donchian_high_entry[i] = np.max(high[i-lookback_entry:i])
        donchian_low_entry[i] = np.min(low[i-lookback_entry:i])
    
    for i in range(lookback_exit, n):
        donchian_high_exit[i] = np.max(high[i-lookback_exit:i])
        donchian_low_exit[i] = np.min(low[i-lookback_exit:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback_entry, 50, 20)  # Donchian(20), EMA50, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(donchian_high_entry[i]) or np.isnan(donchian_low_entry[i]) or
            np.isnan(donchian_high_exit[i]) or np.isnan(donchian_low_exit[i])):
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
        vol_confirm = curr_volume > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Donchian(20) high
                if curr_high > donchian_high_entry[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below Donchian(20) low
                if curr_low < donchian_low_entry[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below Donchian(10) low
            if curr_low < donchian_low_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian(10) high
            if curr_high > donchian_high_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0