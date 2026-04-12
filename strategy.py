#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily typical price
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate ATR(5) on daily data for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    
    # Calculate Camarilla levels for today based on yesterday's data
    # H4 = Close + 1.1/2 * (High - Low) = Close + 0.55 * Range
    # L4 = Close - 1.1/2 * (High - Low) = Close - 0.55 * Range
    # H3 = Close + 1.1/4 * (High - Low) = Close + 0.275 * Range
    # L3 = Close - 1.1/4 * (High - Low) = Close - 0.275 * Range
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 0.55 * range_1d
    camarilla_l4 = close_1d - 0.55 * range_1d
    camarilla_h3 = close_1d + 0.275 * range_1d
    camarilla_l3 = close_1d - 0.275 * range_1d
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume filter - 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Choppiness regime filter on 1d timeframe
    # Calculate Choppiness Index: higher = ranging, lower = trending
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    # Only trade in trending markets (Chop < 38.2) or strong breaks in ranging markets
    chop_trending = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout signals with volume confirmation
        # Long: price breaks above H4 level with volume
        long_signal = close[i] > camarilla_h4_aligned[i] and volume_ok[i]
        # Short: price breaks below L4 level with volume
        short_signal = close[i] < camarilla_l4_aligned[i] and volume_ok[i]
        
        # Exit when price returns to H3/L3 levels
        exit_long = close[i] < camarilla_h3_aligned[i]
        exit_short = close[i] > camarilla_l3_aligned[i]
        
        # Execute trades - only in trending regime or with strong volume confirmation
        if long_signal and position != 1 and (chop_trending[i] or volume[i] > 2 * vol_ma[i]):
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1 and (chop_trending[i] or volume[i] > 2 * vol_ma[i]):
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals