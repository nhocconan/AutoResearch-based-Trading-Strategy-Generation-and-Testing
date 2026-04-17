#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe strategy using 1-day pivot points with volume confirmation and trend filter.
Long when price breaks above R1 with volume > 1.5x 20-bar volume MA and price > 20-bar EMA.
Short when price breaks below S1 with volume > 1.5x 20-bar volume MA and price < 20-bar EMA.
Exit when price returns to pivot point (PP) or volume drops below average.
Designed for 6h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years.
Uses 1-day pivot points calculated from prior day's OHLC.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data once before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate pivot points from prior day's OHLC
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pp - df_1d['low']
    s1 = 2 * pp - df_1d['high']
    
    # Align pivot levels to 6h timeframe (wait for daily bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # 20-bar EMA for trend filter
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 20-period volume MA
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_20[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and uptrend
            if price > r1_val and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and downtrend
            elif price < s1_val and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to pivot point or volume drops
            if price <= pp_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to pivot point or volume drops
            if price >= pp_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_Pivot_R1S1_Volume_EMA_Filter"
timeframe = "6h"
leverage = 1.0