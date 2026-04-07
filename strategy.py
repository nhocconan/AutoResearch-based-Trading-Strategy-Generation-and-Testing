#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot levels from daily data + volume spike + choppiness regime filter
# Long when price touches or crosses above L3 Camarilla level with volume > 1.5x average and chop > 61.8 (range)
# Short when price touches or crosses below H3 Camarilla level with volume > 1.5x average and chop > 61.8
# Exit when price moves to H4/L4 levels or chop < 38.2 (trending)
# Uses daily Camarilla levels for structure and 4h chop to identify ranging markets
# Position size: 0.25, designed for mean reversion in ranging markets with volatility filters

name = "4h_camarilla_daily_chop_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    range_prev = high_prev - low_prev
    camarilla_h5 = close_prev + range_prev * 1.1 / 2
    h4 = close_prev + range_prev * 1.1
    h3 = close_prev + range_prev * 1.1 * 5/12
    l3 = close_prev - range_prev * 1.1 * 5/12
    l4 = close_prev - range_prev * 1.1
    l5 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 4h Choppiness Index (CHOP) for regime detection
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    close_4h = get_htf_data(prices, '4h')['close'].values
    
    # True Range for CHOP calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), chop)
    
    # 4h volume average for confirmation
    volume_4h = get_htf_data(prices, '4h')['volume'].values
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_4h_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price reaches L4 or chop < 38.2 (trending)
            if close[i] <= l4_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches H4 or chop < 38.2 (trending)
            if close[i] >= h4_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price near L3/H3 with volume spike and chop > 61.8 (range)
            # Long: price crosses above L3, volume confirmation, ranging market
            if (close[i] > l3_aligned[i] and 
                volume[i] > 1.5 * volume_ma_4h_aligned[i] and
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below H3, volume confirmation, ranging market
            elif (close[i] < h3_aligned[i] and
                  volume[i] > 1.5 * volume_ma_4h_aligned[i] and
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
    
    return signals