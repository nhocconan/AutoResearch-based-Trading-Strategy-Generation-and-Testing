#!/usr/bin/env python3
"""
Hypothesis: 6h weekly Camarilla pivot reversal with volume confirmation and daily VWAP filter.
Long when price touches weekly S3 with volume > 1.5x average and close > daily VWAP;
Short when price touches weekly R3 with volume > 1.5x average and close < daily VWAP.
Exit when price returns to weekly pivot or 1.5x ATR stop. Weekly Camarilla levels (S3/R3) represent
strong institutional support/resistance where reversals often occur. Volume surge confirms institutional
participation. Designed for 15-25 trades/year to minimize fee drag. Works in ranging markets via
mean reversion at extremes and in trending markets via continuation after pullbacks to S3/R3.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla pivot levels
    range_1w = high_1w - low_1w
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # S3 and R3 are the key reversal levels
    s3 = close_1w - range_1w * 1.1 / 2.0
    r3 = close_1w + range_1w * 1.1 / 2.0
    # Pivot for exit
    pivot_val = pivot
    
    # Align weekly Camarilla levels to 6h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_val)
    
    # Load daily data ONCE before loop for VWAP filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily VWAP (typical price * volume cumulative)
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_num = np.cumsum(typical_price * df_1d['volume'].values)
    vwap_den = np.cumsum(df_1d['volume'].values)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ATR for stop (14-period on 6h)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current daily volume aligned to 6h
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_1d_current = vol_1d_aligned[i]
        
        if position == 0:
            # Enter long: touch weekly S3 with volume surge and close > daily VWAP
            if (price_low <= s3_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                price_close > vwap_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: touch weekly R3 with volume surge and close < daily VWAP
            elif (price_high >= r3_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                  price_close < vwap_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: return to weekly pivot or 1.5x ATR stop
            exit_signal = False
            
            if position == 1:
                # Exit long: touch weekly pivot OR price < entry - 1.5*ATR
                if price_low <= pivot_aligned[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use S3 as entry level for long
                    entry_level = s3_aligned[i-1] if i >= 1 else s3_aligned[0]
                    if price_close < entry_level - 1.5 * atr[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: touch weekly pivot OR price > entry + 1.5*ATR
                if price_high >= pivot_aligned[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use R3 as entry level for short
                    entry_level = r3_aligned[i-1] if i >= 1 else r3_aligned[0]
                    if price_close > entry_level + 1.5 * atr[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyCamarilla_S3_R3_Reversal_1dVWAP_Volume1.5x"
timeframe = "6h"
leverage = 1.0