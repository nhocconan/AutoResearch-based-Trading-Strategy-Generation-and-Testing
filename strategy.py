#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_ChoppinessFilter_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter, volume spike (>2x avg volume), and choppiness regime filter (CHOP > 61.8 = range = mean revert, CHOP < 38.2 = trending = trend follow). Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-100 trades over 4 years (12-25/year) on 4h timeframe. Choppiness filter avoids whipsaws in sideways markets, improving performance in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and chop
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
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) on 4h data
    # CHOP = 100 * log10(sum(ATR(1) over 14) / (log10(HH(14) - LL(14)) / log10(14)))
    tr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = hh14 - ll14
    chop = 100 * np.log10(sum_atr1 * np.log10(14) / range14) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((range14 > 0) & (~np.isnan(sum_atr1)), chop, 50.0)  # default to neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Camarilla calculation, 34 for EMA, 14 for chop)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Need previous day's OHLC for Camarilla levels
        if i < 1:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous period's high, low, close (for Camarilla calculation)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position if invalid range
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R3 and S3 levels (stronger levels)
        r3 = prev_close + range_val * 1.1 / 4
        s3 = prev_close - range_val * 1.1 / 4
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        chop_val = chop[i]
        
        # Skip if any data not ready
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(chop_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.0x average volume (stricter for fewer trades)
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Choppiness regime filter: CHOP < 38.2 = trending (trend follow), CHOP > 61.8 = range (mean revert)
        # We use CHOP < 38.2 for trend following breakouts
        chop_filter = chop_val < 38.2
        
        # Long logic: price breaks above R3 with 1d uptrend, volume confirmation, and trending regime
        long_condition = (close_val > r3) and (close_val > ema_val) and volume_confirmed and chop_filter
        # Short logic: price breaks below S3 with 1d downtrend, volume confirmation, and trending regime
        short_condition = (close_val < s3) and (close_val < ema_val) and volume_confirmed and chop_filter
        
        # Exit logic: trend reversal or opposite Camarilla level break or chop becomes too high (range)
        exit_long = (close_val < ema_val) or (close_val < s3) or (chop_val > 61.8)
        exit_short = (close_val > ema_val) or (close_val > r3) or (chop_val > 61.8)
        
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

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_ChoppinessFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0