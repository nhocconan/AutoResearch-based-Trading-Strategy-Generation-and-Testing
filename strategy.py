#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + ATR-based stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5 * 4h volume MA(20).
- Exit: Opposite Donchian breakout (Long exits when price < Donchian(10) low, Short exits when price > Donchian(10) high).
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian channels provide structure, EMA34 filters trend direction, volume confirms conviction.
- Works in bull (buying breakouts) and bear (selling breakdowns) with regime filter reducing whipsaws.
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
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume MA(20)
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    # Donchian high = max(high, lookback)
    # Donchian low = min(low, lookback)
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
    start_idx = max(lookback_entry, 34, 20)  # Donchian(20), EMA34, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
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
        
        # Trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
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

name = "4h_Donchian20_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0