#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d: fade at R3/S3 with volume confirmation and chop filter
# - Long: price < S3, volume > 1.5x 20-period avg, CHOP(14) > 61.8 (range), closes above S3
# - Short: price > R3, volume > 1.5x 20-period avg, CHOP(14) > 61.8 (range), closes below R3
# - Exit: price returns to opposite Camarilla level (R3 for long exit, S3 for short exit)
# - Uses 1d Camarilla levels calculated from prior 1d OHLC, aligned to 12h
# - Works in ranging markets (chop > 61.8) by fading extremes at Camarilla levels
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "12h_1d_camarilla_fade_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on prior 1d OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous period's range
    # R4 = close + 1.1*(high-low)*1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # S4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_s4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use prior 1d period's levels for current 12h bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (highest_high - lowest_low)) / log10(14)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First bar TR
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop = pd.Series(chop_raw).rolling(window=1, min_periods=1).mean().values  # Just to handle NaN
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Regime filter: Choppiness Index > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        # Price position relative to Camarilla levels
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        # 12h bar close position relative to Camarilla levels
        # For long: we want price to close above S3 after being below it
        # For short: we want price to close below R3 after being above it
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long fade: price < S3, volume confirmation, chop filter, closes above S3
        if low_price < s3 and vol_confirm and chop_filter and close_price > s3:
            enter_long = True
        
        # Short fade: price > R3, volume confirmation, chop filter, closes below R3
        if high_price > r3 and vol_confirm and chop_filter and close_price < r3:
            enter_short = True
        
        # Exit conditions: mean reversion at opposite Camarilla level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price rises back to R3 (opposite level)
            exit_long = close_price >= r3
        elif position == -1:
            # Exit short if price drops back to S3 (opposite level)
            exit_short = close_price <= s3
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals