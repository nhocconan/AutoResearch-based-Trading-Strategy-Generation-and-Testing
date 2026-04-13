#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
    # Long: price breaks above H3 AND 12h volume > 1.5x avg AND chop < 61.8 (trending)
    # Short: price breaks below L3 AND 12h volume > 1.5x avg AND chop < 61.8 (trending)
    # Exit: price returns to Pivot level or chop > 61.8 (range)
    # Using 4h timeframe for optimal trade frequency, Camarilla for structure,
    # 12h volume for confirmation, chop regime to avoid false breakouts in ranging markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h volume moving average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        vol_ma_12h[i] = np.mean(vol_12h[i-20:i])
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, H4, L3, L4, Pivot
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # Pivot = (high + low + close)/3
    rng = high_1d - low_1d
    h3_1d = close_1d + 1.1 * rng / 2
    l3_1d = close_1d - 1.1 * rng / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 4h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate 4h Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest(high,n) - lowest(low,n))))
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    lookback_period = 14
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate CHOP
    chop = np.full(n, np.nan)
    for i in range(lookback_period, n):
        if not np.isnan(atr[i-lookback_period+1:i+1]).any():
            atr_sum = np.nansum(atr[i-lookback_period+1:i+1])
            highest_high = np.nanmax(high[i-lookback_period+1:i+1])
            lowest_low = np.nanmin(low[i-lookback_period+1:i+1])
            if highest_high > lowest_low and atr_sum > 0:
                chop[i] = 100 * np.log10(atr_sum) / (np.log10(lookback_period) * np.log10((highest_high - lowest_low) / atr_sum))
            else:
                chop[i] = 50.0  # neutral when no range
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    volume_conf = volume > (1.5 * vol_ma_12h_aligned)
    
    # Chop regime filter: CHOP < 61.8 = trending (favor breakouts)
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_conf[i]) or 
            np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic: Camarilla breakout + volume confirmation + trending regime
        long_entry = (close[i] > h3_1d_aligned[i]) and volume_conf[i] and trending_regime[i]
        short_entry = (close[i] < l3_1d_aligned[i]) and volume_conf[i] and trending_regime[i]
        
        # Exit logic: return to pivot or chop > 61.8 (range)
        long_exit = (close[i] < pivot_1d_aligned[i]) or (~trending_regime[i])
        short_exit = (close[i] > pivot_1d_aligned[i]) or (~trending_regime[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0