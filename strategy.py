#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Filter_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 12h data for choppiness regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate choppiness index on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Sum of true range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(tr_sum / (hh - ll)) / log10(14)
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    
    # Chop > 61.8 = ranging market (mean revert)
    # Chop < 38.2 = trending market (trend follow)
    chop_12h = chop
    
    # Align all to 4h
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    chop_12h_4h = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Donchian channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_4h[i]) or np.isnan(chop_12h_4h[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34 = ema34_1d_4h[i]
        chop_val = chop_12h_4h[i]
        upper_ch = upper[i]
        lower_ch = lower[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above upper band in trending market (chop < 38.2) with volume
            if close[i] > upper_ch and chop_val < 38.2 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band in trending market (chop < 38.2) with volume
            elif close[i] < lower_ch and chop_val < 38.2 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower band OR chop > 61.8 (range) with mean reversion
            if close[i] < lower_ch or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper band OR chop > 61.8 (range) with mean reversion
            if close[i] > upper_ch or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals