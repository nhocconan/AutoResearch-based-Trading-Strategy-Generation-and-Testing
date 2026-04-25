#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_Volume_Chop
Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (EMA34), volume confirmation, and choppiness regime filter. 
Long when price breaks above R3 with volume spike and trending market (CHOP < 38.2), short when breaks below S3 with volume spike and trending market. 
Uses discrete sizing (0.25) to minimize fees. Designed for 20-50 trades/year, works in bull markets via breakouts and in bear markets via breakdowns with trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF Camarilla pivot, EMA trend, and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for 1d (based on previous day)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3 and S3 as primary breakout levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan  # First value has no previous
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_s3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (ATR14 period high-low range)) / log10(14)
    tr1_1d = np.maximum(np.abs(high_1d[1:] - low_1d[:-1]), 
                        np.maximum(np.abs(high_1d[1:] - prev_close_1d[:-1]),
                                   np.abs(low_1d[1:] - prev_close_1d[:-1])))
    tr1_1d = np.concatenate([[np.nan], tr1_1d])  # Align with original arrays
    
    atr14_1d = pd.Series(tr1_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr14_1d = pd.Series(atr14_1d).rolling(window=14, min_periods=14).sum().values
    range14_1d = hh14_1d - ll14_1d
    chop_1d = 100 * np.log10(sum_atr14_1d / range14_1d) / np.log10(14)
    chop_1d = np.where(range14_1d > 0, chop_1d, 100)  # Avoid division by zero
    
    # Align HTF indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h volume confirmation (volume > 1.5x 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R3, volume spike, trending market (CHOP < 38.2), price above EMA34 (bullish bias)
            long_signal = (close[i] > camarilla_r3_aligned[i]) and volume_spike[i] and (chop_aligned[i] < 38.2) and (close[i] > ema34_aligned[i])
            # Short: price breaks below S3, volume spike, trending market (CHOP < 38.2), price below EMA34 (bearish bias)
            short_signal = (close[i] < camarilla_s3_aligned[i]) and volume_spike[i] and (chop_aligned[i] < 38.2) and (close[i] < ema34_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price closes below S3 (breakdown) or chop becomes too high (choppy market)
            exit_signal = (close[i] < camarilla_s3_aligned[i]) or (chop_aligned[i] > 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above R3 (breakout) or chop becomes too high (choppy market)
            exit_signal = (close[i] > camarilla_r3_aligned[i]) or (chop_aligned[i] > 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0