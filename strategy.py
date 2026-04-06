#!/usr/bin/env python3
"""
12h Camarilla pivot with 1d trend filter and volume confirmation
Hypothesis: Camarilla pivot levels on daily timeframe act as strong support/resistance.
Price reversals at these levels with 1d trend alignment and volume confirmation provide
high-probability entries. Works in bull (buy at S1/S2 in uptrend) and bear (sell at R1/R2 in downtrend).
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_r2 = np.full(len(close_1d), np.nan)
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    camarilla_s2 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            camarilla_r4[i] = prev_close + range_ * 1.1 / 2
            camarilla_r3[i] = prev_close + range_ * 1.1 / 4
            camarilla_r2[i] = prev_close + range_ * 1.1 / 6
            camarilla_r1[i] = prev_close + range_ * 1.1 / 12
            camarilla_s1[i] = prev_close - range_ * 1.1 / 12
            camarilla_s2[i] = prev_close - range_ * 1.1 / 6
            camarilla_s3[i] = prev_close - range_ * 1.1 / 4
            camarilla_s4[i] = prev_close - range_ * 1.1 / 2
    
    # Get 1d data for trend filter (EMA50)
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d data to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for Camarilla and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 12h volume > 1.5x 1d average volume
        vol_threshold = vol_ma_1d_aligned[i] * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses S1 (stop reversal) or against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < camarilla_s1_aligned[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price crosses R1 (stop reversal) or against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > camarilla_r1_aligned[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Mean reversion entries at Camarilla levels with 1d trend
                # Long: price at S1/S2 in uptrend with volume
                near_s1 = abs(close[i] - camarilla_s1_aligned[i]) < (0.1 * atr[i])
                near_s2 = abs(close[i] - camarilla_s2_aligned[i]) < (0.15 * atr[i])
                long_setup = (near_s1 or near_s2) and trend_1d_aligned[i] == 1 and volume_filter
                
                # Short: price at R1/R2 in downtrend with volume
                near_r1 = abs(close[i] - camarilla_r1_aligned[i]) < (0.1 * atr[i])
                near_r2 = abs(close[i] - camarilla_r2_aligned[i]) < (0.15 * atr[i])
                short_setup = (near_r1 or near_r2) and trend_1d_aligned[i] == -1 and volume_filter
                
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals