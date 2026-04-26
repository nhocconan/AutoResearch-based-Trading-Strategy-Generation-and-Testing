#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakouts on 4h with 1d EMA34 trend filter, volume confirmation (>2x average), and choppiness regime filter (CHOP > 61.8 for mean reversion avoidance). 
In trending markets (CHOP < 38.2): breakouts continue trend. In ranging markets (CHOP > 61.8): fade breakouts at extreme levels. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 trades over 4 years (19-50/year) on 4h timeframe.
Designed to work in both bull and bear markets via trend filter and regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 1d data for choppiness index calculation
    df_1d_chop = get_htf_data(prices, '1d')
    if len(df_1d_chop) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d_chop['high'].values
    low_1d = df_1d_chop['low'].values
    close_1d = df_1d_chop['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d bars
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (hh_14 - ll_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d_chop, chop)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 1 for Camarilla, 34 for EMA, 14 for CHOP)
    start_idx = max(1, 34, 14)
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using previous day's OHLC
        # For 4h timeframe, previous day = previous 6 bars
        prev_1d_idx = i - 6
        if prev_1d_idx < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        prev_high = high[prev_1d_idx]
        prev_low = low[prev_1d_idx]
        prev_close = close[prev_1d_idx]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R3 and S3 levels
        R3 = prev_close + (range_val * 1.1 / 4)
        S3 = prev_close - (range_val * 1.1 / 4)
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(R3) or np.isnan(S3) or np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(chop_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Regime filters
        chop_trending = chop_val < 38.2  # Trending regime
        chop_ranging = chop_val > 61.8   # Ranging regime
        
        # Long logic: 
        # In trending markets: break above R3 with trend and volume
        # In ranging markets: fade at S3 (mean reversion) with volume
        long_trending = (close_val > R3) and (close_val > ema_val) and volume_confirmed and chop_trending
        long_ranging = (close_val < S3) and (close_val < ema_val) and volume_confirmed and chop_ranging
        long_condition = long_trending or long_ranging
        
        # Short logic:
        # In trending markets: break below S3 with trend and volume
        # In ranging markets: fade at R3 (mean reversion) with volume
        short_trending = (close_val < S3) and (close_val < ema_val) and volume_confirmed and chop_trending
        short_ranging = (close_val > R3) and (close_val > ema_val) and volume_confirmed and chop_ranging
        short_condition = short_trending or short_ranging
        
        # Exit logic: trend reversal or opposite extreme
        exit_long = (close_val < ema_val) or (close_val < S3) or (close_val > R3 and chop_ranging)
        exit_short = (close_val > ema_val) or (close_val > R3) or (close_val < S3 and chop_ranging)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0