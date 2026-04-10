#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volume confirmation and chop regime filter
# - Long when price breaks above Donchian upper band (20) AND 1d ATR ratio > 1.3 (expanding volatility) AND chop < 61.8 (trending market)
# - Short when price breaks below Donchian lower band (20) AND 1d ATR ratio > 1.3 AND chop < 61.8
# - Exit when price returns to Donchian middle band (20-period average)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts work in both bull and bear markets when combined with volatility expansion and trend filter
# - ATR ratio filter ensures we trade during genuine volatility expansion, not chop
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_atr_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 4h Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    # Calculate True Range
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]  # First bar
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # Calculate ATR (14-period) using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index
    hh = rolling_max(high, 14)
    ll = rolling_min(low, 14)
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if hh[i] > ll[i]:
            log_sum = np.log10(rolling_sum(tr, 14)[i] / (hh[i] - ll[i]))
            chop[i] = 100 * log_sum / np.log10(14)
        else:
            chop[i] = 50.0
    
    chop_regime = chop < 61.8  # Trending market
    
    # Pre-compute 1d ATR ratio (current ATR / 20-period average ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    # Calculate 1d ATR (14-period)
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:15])
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 20-period average of 1d ATR
    atr_ma_1d = np.zeros_like(atr_1d)
    for i in range(19, len(atr_1d)):
        atr_ma_1d[i] = np.mean(atr_1d[i-19:i+1])
    
    # ATR ratio: current ATR / average ATR (values > 1 indicate expanding volatility)
    atr_ratio_1d = np.ones_like(atr_1d)
    valid_ma = (atr_ma_1d > 0) & ~np.isnan(atr_ma_1d)
    atr_ratio_1d[valid_ma] = atr_1d[valid_ma] / atr_ma_1d[valid_ma]
    
    # Align HTF indicators to 4h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(chop_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND ATR ratio > 1.3 AND chop regime (trending)
            if (close[i] > donchian_high[i] and 
                atr_ratio_1d_aligned[i] > 1.3 and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND ATR ratio > 1.3 AND chop regime (trending)
            elif (close[i] < donchian_low[i] and 
                  atr_ratio_1d_aligned[i] > 1.3 and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Donchian middle band (mean reversion)
            exit_long = (position == 1 and close[i] < donchian_mid[i])
            exit_short = (position == -1 and close[i] > donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals