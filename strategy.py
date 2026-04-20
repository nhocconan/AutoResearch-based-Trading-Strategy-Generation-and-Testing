#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Donchian20_WeeklyTrend_ForceIndex"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Donchian Channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 20-period rolling high/low
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === Weekly Force Index (13-period EMA of price change * volume) ===
    # Price change
    price_change = np.diff(close_1w, prepend=close_1w[0])
    # Force Index raw
    force_raw = price_change * df_1w['volume'].values
    # 13-period EMA of Force Index
    force_series = pd.Series(force_raw)
    force_index = force_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Align Force Index to 6h
    force_index_aligned = align_htf_to_ltf(prices, df_1w, force_index)
    
    # === 6x Force Index Thresholds for Trend Strength ===
    force_ma = pd.Series(force_index_aligned).ewm(span=6, min_periods=6, adjust=False).mean().values
    force_std = pd.Series(force_index_aligned).rolling(window=6, min_periods=6).std().values
    force_upper = force_ma + 1.5 * force_std
    force_lower = force_ma - 1.5 * force_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        dh_val = donchian_high_aligned[i]
        dl_val = donchian_low_aligned[i]
        fi_val = force_index_aligned[i]
        fu_val = force_upper[i]
        fl_val = force_lower[i]
        
        # Skip if any value is NaN
        if (np.isnan(dh_val) or np.isnan(dl_val) or 
            np.isnan(fi_val) or np.isnan(fu_val) or np.isnan(fl_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly Donchian high with strong upward force
            if close_val > dh_val and fi_val > fu_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian low with strong downward force
            elif close_val < dl_val and fi_val < fl_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below weekly Donchian low OR force turns negative
            if close_val < dl_val or fi_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above weekly Donchian high OR force turns positive
            if close_val > dh_val or fi_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals