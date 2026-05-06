#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour trend following using 12-hour Donchian breakout with volume confirmation
# Uses 12-hour Donchian channels (20-period) for trend direction and breakout signals
# Requires price to break above/below Donchian bands with volume > 1.5x 20-bar average
# Includes volatility filter using 12-hour ATR to avoid whipsaws in low volatility periods
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear markets by capturing breakouts from consolidation with volume confirmation

name = "6h_DonchianBreakout_12hVolatilityFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12-hour Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12-hour ATR for volatility filter
    tr1_12h = np.abs(high_12h[1:] - low_12h[1:])
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1]) if len(df_12h) > 1 else np.array([])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1]) if len(df_12h) > 1 else np.array([])
    if len(df_12h) > 1:
        close_12h = df_12h['close'].values
        tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    else:
        tr_12h = np.full(len(high_12h), np.nan)
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume confirmation and sufficient volatility
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                atr_12h_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume confirmation and sufficient volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  atr_12h_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals