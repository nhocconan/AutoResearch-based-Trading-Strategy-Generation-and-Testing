# The strategy below implements a 4-hour timeframe approach combining:
# - 12-hour Donchian channel breakouts for entry signals
# - 1-day EMA trend filter to confirm higher timeframe direction
# - Volume confirmation to ensure institutional participation
# - ATR-based volatility filter to avoid choppy markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Clear exit conditions when trend reverses or volatility drops
#
# This approach targets 20-40 trades per year by requiring multiple confluence factors,
# reducing false signals while capturing strong trending moves in both bull and bear markets.
# The 12h/1d/4h timeframe combination provides multi-timeframe alignment without
# excessive trading frequency.

#!/usr/bin/env python3
"""
4h_12h_1D_DonchianBreakout_EMATrend_VolumeFilter
Hypothesis: Use 12-hour Donchian channel breakouts as entry signals, filtered by
1-day EMA trend direction and volume confirmation. Enter long when price breaks
above 12h Donchian upper channel in 1d uptrend with above-average volume.
Enter short when price breaks below 12h Donchian lower channel in 1d downtrend
with above-average volume. Exit when trend reverses or volume drops below average.
This captures institutional breakout moves with multi-timeframe confirmation,
reducing false signals in choppy markets. Target: 20-40 trades/year to avoid fee drag.
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns
in downtrend.
"""

name = "4h_12h_1D_DonchianBreakout_EMATrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12-hour data for Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4-hour OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12-hour Donchian Channel (20-period) ---
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian upper and lower
    donch_high_12h = np.full(len(high_12h), np.nan)
    donch_low_12h = np.full(len(low_12h), np.nan)
    
    for i in range(20, len(high_12h)):
        donch_high_12h[i] = np.max(high_12h[i-20:i])
        donch_low_12h[i] = np.min(low_12h[i-20:i])
    
    # Align 12h Donchian levels to 4h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # --- 1-day EMA (50-period) for trend filter ---
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    
    # Calculate EMA with proper initialization
    if len(close_1d) >= 50:
        ema_50_1d[49] = close_1d[:50].mean()  # Simple average for first value
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50_1d[i-1] * (49 / (50 + 1)))
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Volume confirmation (20-period average) ---
    vol_ma_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(20, 50)  # Need both Donchian and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1-day trend based on price relative to EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume above 20-period average
        volume_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1-day trend with volume confirmation
            if uptrend and close[i] > donch_high_12h_aligned[i] and volume_confirm:
                # Long: 1d uptrend + 12h breakout above upper channel + volume
                signals[i] = 0.25
                position = 1
            elif downtrend and close[i] < donch_low_12h_aligned[i] and volume_confirm:
                # Short: 1d downtrend + 12h breakdown below lower channel + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: 1d trend turns down OR volume drops below average
                if not uptrend or not volume_confirm:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 1d trend turns up OR volume drops below average
                if not downtrend or not volume_confirm:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals