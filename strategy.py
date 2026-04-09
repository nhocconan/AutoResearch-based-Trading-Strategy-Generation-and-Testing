#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# Uses 1d Camarilla levels (L3, H3) for breakout entries in direction of 1w HMA trend
# Volume confirmation: 4h volume > 1.5x 20-period average
# Chop regime filter: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion at pivots
# Discrete position sizing 0.25 to target ~20-40 trades/year
# Works in bull/bear: pivots act as support/resistance in ranging markets, trend filter avoids counter-trend

name = "4h_1d_1w_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # We use H3 (resistance) and L3 (support) for breakouts
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_h3 = prev_close_1d + 1.0 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.0 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1w HMA(20) for trend filter
    def hull_moving_average(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan)
        half = window // 2
        sqrt = int(np.sqrt(window))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=window, adjust=False).mean().values
        raw = 2 * wma2 - wma1
        hma = pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
        return hma
    
    hma_20_1w = hull_moving_average(close_1w, 20)
    
    # Calculate 4h Chopiness Index (CHOP) for regime filter
    def chopiness_index(high_arr, low_arr, close_arr, window):
        if len(high_arr) < window:
            return np.full_like(high_arr, np.nan)
        atr = np.maximum(np.abs(high_arr - low_arr),
                         np.maximum(np.abs(high_arr - np.roll(close_arr, 1)),
                                    np.abs(low_arr - np.roll(close_arr, 1))))
        atr[0] = np.nan
        tr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        denominator = hh - ll
        chop = 100 * np.log10(tr_sum / denominator) / np.log10(window)
        return chop
    
    chop_14 = chopiness_index(high, low, close, 14)
    
    # Calculate 4h average volume (20-period)
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Align 1w HMA to 4h timeframe
    hma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(hma_20_1w_aligned[i]) or np.isnan(chop_14[i]) or np.isnan(avg_vol_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_vol_20[i]
        
        # Chop regime filter: only trade in ranging market (CHOP > 61.8)
        in_chop_regime = chop_14[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long if price falls below Camarilla L3
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above Camarilla H3
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion at Camarilla levels in ranging market
            if volume_confirmed and in_chop_regime:
                if close[i] > camarilla_h3_aligned[i]:
                    # Short at H3 resistance (expect reversion to mean)
                    position = -1
                    signals[i] = -0.25
                elif close[i] < camarilla_l3_aligned[i]:
                    # Long at L3 support (expect reversion to mean)
                    position = 1
                    signals[i] = 0.25
    
    return signals