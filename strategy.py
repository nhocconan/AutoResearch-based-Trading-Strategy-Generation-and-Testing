#!/usr/bin/env python3
# 6h_1w_donchian_pivot_v1
# Hypothesis: 6h strategy using weekly Donchian channels (20) for trend direction and daily Camarilla pivots for entry timing.
# Goes long when price breaks above weekly Donchian high AND closes above daily H3 pivot with volume confirmation.
# Goes short when price breaks below weekly Donchian low AND closes below daily L3 pivot with volume confirmation.
# Uses ATR filter to avoid low volatility whipsaws. Discrete sizing (±0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via weekly structure and daily mean reversion at pivots.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data ONCE before loop (for Donchian)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily HTF data ONCE before loop (for Camarilla pivots)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: max high over last 20 weekly candles
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: min low over last 20 weekly candles
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe (completed HTF candle only)
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels for daily
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align daily Camarilla levels to 6h timeframe (completed HTF candle only)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume spike detection (20-period volume average on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    # ATR filter for volatility (14-period ATR on 6h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: use only high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_filter = atr > 0  # Ensure ATR is valid (non-zero)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below daily L3 level OR weekly Donchian low breaks
            if (close[i] < l3_1d_aligned[i]) or (close[i] < donchian_low_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above daily H3 level OR weekly Donchian high breaks
            if (close[i] > h3_1d_aligned[i]) or (close[i] > donchian_high_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high AND closes above daily H3 with volume spike
            if (close[i] > donchian_high_1w_aligned[i]) and (close[i] > h3_1d_aligned[i]) and vol_spike[i] and atr_filter[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low AND closes below daily L3 with volume spike
            elif (close[i] < donchian_low_1w_aligned[i]) and (close[i] < l3_1d_aligned[i]) and vol_spike[i] and atr_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals