#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v26"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    r1 = close_1d + range_hl * 1.1 / 12
    r2 = close_1d + range_hl * 1.1 / 6
    r3 = close_1d + range_hl * 1.1 / 4
    r4 = close_1d + range_hl * 1.1 / 2
    s1 = close_1d - range_hl * 1.1 / 12
    s2 = close_1d - range_hl * 1.1 / 6
    s3 = close_1d - range_hl * 1.1 / 4
    s4 = close_1d - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 1w data for trend filter (long-term trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_10_1w = pd.Series(close_1w).rolling(window=10, min_periods=10).mean().values
    sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
    
    # Volume filter - 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Choppiness regime filter (4h)
    # Choppiness Index: high values = ranging, low values = trending
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (atr * 14)) / np.log10(14)
    chop[np.isnan(chop) | (chop < 0)] = 50  # handle edge cases
    
    # Chop > 50 = ranging/choppy, Chop < 50 = trending
    chopping_market = chop > 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(sma_10_1w_aligned[i]) or np.isnan(volume_ok[i]) or np.isnan(chopping_market[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend from weekly SMA
        uptrend = close[i] > sma_10_1w_aligned[i]
        downtrend = close[i] < sma_10_1w_aligned[i]
        
        # Camarilla breakout signals with volume and chop filter
        # Long: break above R3 in trending market OR bounce from S3 in ranging market
        long_signal = False
        if chopping_market[i]:  # ranging market - mean reversion
            long_signal = (close[i] < s3_aligned[i] * 1.005) and volume_ok[i]  # near S3 support
        else:  # trending market - breakout
            long_signal = (close[i] > r3_aligned[i] * 1.005) and volume_ok[i] and uptrend
        
        # Short: break below S3 in trending market OR bounce from R3 in ranging market
        short_signal = False
        if chopping_market[i]:  # ranging market - mean reversion
            short_signal = (close[i] > r3_aligned[i] * 0.995) and volume_ok[i]  # near R3 resistance
        else:  # trending market - breakout
            short_signal = (close[i] < s3_aligned[i] * 0.995) and volume_ok[i] and downtrend
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if chopping_market[i]:  # ranging market - exit at opposite levels
            exit_long = close[i] > r3_aligned[i] * 0.995  # hit resistance
            exit_short = close[i] < s3_aligned[i] * 1.005  # hit support
        else:  # trending market - trail with stop or reverse
            exit_long = close[i] < s3_aligned[i] * 1.005  # broke support
            exit_short = close[i] > r3_aligned[i] * 0.995  # broke resistance
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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