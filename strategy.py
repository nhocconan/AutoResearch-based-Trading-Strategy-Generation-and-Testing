#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter
# Camarilla pivots from 1d provide intraday support/resistance that work in ranging markets
# Volume spike (>2.0x average) confirms institutional participation at pivot levels
# Choppiness index (CHOP > 61.8) ensures we only trade in ranging regimes where mean reversion works
# Position size: 0.25 for long/short at pivot touches
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_camarilla_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low), etc.
    # We'll use R3, S3 as primary levels for mean reversion
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First period
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    rangep1d = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + 1.1 * rangep1d
    camarilla_s3 = prev_close_1d - 1.1 * rangep1d
    
    # Calculate 1d Choppiness Index (14-period) for regime filtering
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
    chop_14 = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align Camarilla levels and Chop to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average 12h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: only trade when market is choppy (ranging) - CHOP > 61.8
        regime_filter = chop_aligned[i] > 61.8
        
        if not (volume_confirmed and regime_filter):
            signals[i] = 0.0
            continue
        
        # Fixed position size for mean reversion trades
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price reaches midpoint or R3 level (take profit)
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if close[i] >= midpoint or close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price reaches midpoint or S3 level (take profit)
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if close[i] <= midpoint or close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion trading at Camarilla S3/R3 levels with volume and chop confirmation
            # Long at S3 (support), Short at R3 (resistance)
            if volume_confirmed:
                if close[i] <= s3_aligned[i]:
                    position = 1
                    signals[i] = position_size
                elif close[i] >= r3_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals