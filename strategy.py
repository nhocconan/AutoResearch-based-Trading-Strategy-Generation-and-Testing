#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and chop filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily TRIX (15-period)
    # EMA1: EMA(close, 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2: EMA(EMA1, 15)
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3: EMA(EMA2, 15)
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX: (EMA3 - prev_EMA3) / prev_EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix_raw[0] = 0
    trix_1d = trix_raw
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Daily Choppiness Index (14-period) for regime filter
    # True Range
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = LOG10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / LOG10(14) * 100
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    chop_raw = np.zeros_like(sum_tr_14)
    mask = range_14 > 0
    chop_raw[mask] = (np.log10(sum_tr_14[mask] / range_14[mask]) / np.log10(14)) * 100
    chop_raw[~mask] = 50  # neutral when range is zero
    chop_1d = chop_raw
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 2.5x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        trix = trix_1d_aligned[i]
        chop = chop_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 2.5 * vol_ma
        # Chop filter: only trade in trending markets (Chop < 38.2) or strong reversals in chop (Chop > 61.8)
        # We'll use Chop < 38.2 for trend following, and Chop > 61.8 for mean reversion
        chop_trending = chop < 38.2
        chop_reversal = chop > 61.8
        
        if position == 0:
            # Long: TRIX crosses above zero with volume confirmation in trending market
            # OR TRIX deep oversold with volume in choppy market (mean reversion)
            if (trix > 0 and trix_1d_aligned[i-1] <= 0 and volume_confirmed and chop_trending) or \
               (trix < -1.5 and volume_confirmed and chop_reversal):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume confirmation in trending market
            # OR TRIX overbought with volume in choppy market (mean reversion)
            elif (trix < 0 and trix_1d_aligned[i-1] >= 0 and volume_confirmed and chop_trending) or \
                 (trix > 1.5 and volume_confirmed and chop_reversal):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero OR Chop enters extreme reversal zone
            if trix < 0 or chop > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero OR Chop enters extreme reversal zone
            if trix > 0 or chop > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals