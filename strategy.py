#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + chop regime filter
    # Long: price touches L3 support AND chop > 61.8 (range) AND volume > 1.5x avg
    # Short: price touches H3 resistance AND chop > 61.8 (range) AND volume > 1.5x avg
    # Exit: price moves to opposite H3/L3 level or closes above/below H4/L4
    # Using 12h timeframe for optimal trade frequency (target 12-37/year), Camarilla for structure,
    # daily chop regime to avoid trending markets, and volume confirmation to avoid false signals.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.1*(high - low)
    # L3 = close - 1.1*(high - low)
    # L4 = close - 1.5*(high - low)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align daily Camarilla levels to 12h
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate daily chop regime filter (chop > 61.8 = ranging market)
    # Chop = 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    # Using 14-period chop for daily timeframe
    tr_1d = np.maximum(
        prev_high[1:] - prev_low[1:],
        np.maximum(
            np.abs(prev_high[1:] - prev_close[:-1]),
            np.abs(prev_low[1:] - prev_close[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_14 = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        if not np.isnan(tr_1d[i]):
            if np.isnan(atr_14[i-1]):
                atr_14[i] = np.mean(tr_1d[i-13:i+1])
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate chop: 100 * log10(sum(atr14) / (14 * log10(14))) / log10(14)
    chop = np.full(len(atr_14), np.nan)
    for i in range(14, len(atr_14)):
        if not np.isnan(atr_14[i]):
            sum_atr = np.nansum(atr_14[i-13:i+1])
            if sum_atr > 0:
                chop[i] = 100 * np.log10(sum_atr) / (14 * np.log10(14)) / np.log10(14)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion at pivots)
        ranging_market = chop_1d_aligned[i] > 61.8
        
        # Camarilla pivot touch conditions
        touch_h3 = np.abs(close[i] - h3_1d_aligned[i]) / close[i] < 0.002  # Within 0.2% of H3
        touch_l3 = np.abs(close[i] - l3_1d_aligned[i]) / close[i] < 0.002  # Within 0.2% of L3
        
        # Exit conditions: move to opposite level or break beyond H4/L4
        move_to_h3_from_long = position == 1 and close[i] > h3_1d_aligned[i]
        move_to_l3_from_short = position == -1 and close[i] < l3_1d_aligned[i]
        break_h4 = close[i] > h4_1d_aligned[i]
        break_l4 = close[i] < l4_1d_aligned[i]
        
        # Entry logic: Camarilla touch + ranging market + volume confirmation
        long_entry = touch_l3 and ranging_market and volume_spike[i]
        short_entry = touch_h3 and ranging_market and volume_spike[i]
        
        # Exit logic: move to opposite level or break beyond H4/L4
        long_exit = move_to_l3_from_long or break_l4
        short_exit = move_to_h3_from_short or break_h4
        
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

name = "12h_1d_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0