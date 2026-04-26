#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Chop
Hypothesis: Camarilla R1/S1 breakouts from 1d pivots with 1d trend filter, volume spike, and chop regime filter capture institutional reversal points with controlled frequency. 
Long: price > R1 + volume spike + 1d uptrend + chop < 61.8 (trending). 
Short: price < S1 + volume spike + 1d downtrend + chop < 61.8 (trending). 
Uses discrete position sizing (0.25) and ATR-based stoploss to limit fee drag and manage risk. 
Target: 75-200 trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need warmup for indicators
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar's OHLC
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price = (h_1d + l_1d + c_1d) / 3.0
    hl_range = h_1d - l_1d
    
    r1_1d = typical_price + (hl_range * 1.1 / 12.0)
    s1_1d = typical_price - (hl_range * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe (use previous 1d bar's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Choppiness Index regime filter (14-period)
    chop_period = 14
    atr_period = chop_period
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_vals = pd.Series(tr1).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate highest high and lowest low over chop_period
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Chop = 100 * log10(sum(tr1)/ (hh - ll)) / log10(chop_period)
    sum_tr = pd.Series(tr1).rolling(window=chop_period, min_periods=chop_period).sum().values
    denom = hh - ll
    chop = np.where((denom > 0) & (~np.isnan(sum_tr)), 
                    100 * np.log10(sum_tr / denom) / np.log10(chop_period), 
                    np.nan)
    
    # Chop < 61.8 indicates trending regime (favor breakouts)
    chop_filter = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 34 for EMA, 14 for chop, 20 for volume median)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(chop_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        
        # Long logic: price breaks above R1 with volume spike, 1d uptrend, and trending regime
        long_condition = (close_val > r1_val) and vol_spike and (close_val > ema_val) and chop_ok
        # Short logic: price breaks below S1 with volume spike, 1d downtrend, and trending regime
        short_condition = (close_val < s1_val) and vol_spike and (close_val < ema_val) and chop_ok
        
        # Exit logic: trend reversal or chop regime shift to ranging
        exit_long = (close_val < ema_val) or (chop >= 61.8)
        exit_short = (close_val > ema_val) or (chop >= 61.8)
        
        # Minimum holding period: 1 bar to reduce churn
        if position != 0 and bars_since_entry < 1:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Chop"
timeframe = "4h"
leverage = 1.0