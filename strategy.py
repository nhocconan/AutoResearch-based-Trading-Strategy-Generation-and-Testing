#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChoppyRegime"
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
    
    # Get 1d data for TRIX and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX on 1d close (15-period EMA of EMA of EMA)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    trix_1d = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Choppy index on 1d (14-period)
    hl_range = df_1d['high'].values - df_1d['low'].values
    atr_14 = pd.Series(hl_range).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = max_hh - min_ll
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * np.log10(sum_atr / chop_denom) / np.log10(14)
    
    # Align chop to 4h
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume filter: above 2.0x 12-period average (2 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 14  # Wait for chop calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]
        
        # Session filter: 08-20 UTC
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        # Chop regime: >61.8 = choppy (mean revert), <38.2 = trending
        is_choppy = chop_4h[i] > 61.8
        is_trending = chop_4h[i] < 38.2
        
        if position == 0:
            # Long: TRIX crosses above zero in choppy market (mean reversion bounce)
            if (trix_4h[i] > 0 and trix_4h[i-1] <= 0 and 
                is_choppy and vol_ok and in_session):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero in choppy market (mean reversion fade)
            elif (trix_4h[i] < 0 and trix_4h[i-1] >= 0 and 
                  is_choppy and vol_ok and in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or chop breaks down
            if trix_4h[i] < 0 or not is_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or chop breaks down
            if trix_4h[i] > 0 or not is_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals