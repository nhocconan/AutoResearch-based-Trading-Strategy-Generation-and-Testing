#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w Camarilla pivot levels for trend filter (defines bull/bear regime based on price relative to weekly pivot).
- Entry: Long when price breaks above Donchian(20) high in bull regime (above weekly pivot) with volume > 2.0 * 6h volume MA(20);
         Short when price breaks below Donchian(20) low in bear regime (below weekly pivot) with volume > 2.0 * 6h volume MA(20).
- Exit: Price crosses below Donchian(10) high for long or above Donchian(10) low for short.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian breakouts capture momentum; weekly pivot filter avoids counter-trend trades; volume spike confirms conviction.
- Works in bull (buying breakouts above weekly pivot) and bear (selling breakdowns below weekly pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot (P) = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get 6h data for volume MA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h volume MA(20)
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(5, 20, 20)  # 1w needs 5, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Donchian channels (20-period for entry, 10-period for exit)
        lookback_20 = max(0, i-19)
        donchian_high_20 = np.max(high[lookback_20:i+1])
        donchian_low_20 = np.min(low[lookback_20:i+1])
        
        lookback_10 = max(0, i-9)
        donchian_high_10 = np.max(high[lookback_10:i+1])
        donchian_low_10 = np.min(low[lookback_10:i+1])
        
        # Volume confirmation: 2.0x threshold (strict to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        
        # Trend filter: price relative to weekly pivot
        bull_regime = curr_close > weekly_pivot_aligned[i]
        bear_regime = curr_close < weekly_pivot_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian(20) high in bull regime with volume confirmation
            if curr_high > donchian_high_20 and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low in bear regime with volume confirmation
            elif curr_low < donchian_low_20 and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price crosses below Donchian(10) high
            if curr_close < donchian_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Donchian(10) low
            if curr_close > donchian_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wCamarillaPivot_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0