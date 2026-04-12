#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_trix_volume_chop_v1
# TRIX (12) with volume confirmation and chop regime filter to capture momentum in trending markets.
# TRIX > 0 indicates bullish momentum, TRIX < 0 bearish. Volume confirms institutional participation.
# Chop filter avoids false signals in ranging markets. Works in bull by riding momentum, in bear by
# catching momentum shifts during relief rallies or breakdowns. Target: 20-30 trades/year.
name = "4h_1d_trix_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (12) on daily close
    close_1d = df_1d['close'].values
    # EMA1
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix_raw = np.zeros_like(close_1d)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Align TRIX to 4h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Chop regime filter: avoid choppy markets (CHOP > 61.8)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (atr * np.sqrt(14))) / np.log10(14)
    chop_filter = chop < 61.8  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if TRIX not ready
        if np.isnan(trix_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and chop filters
        if not (vol_confirm[i] and chop_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: TRIX turns positive with volume
        if trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: TRIX turns negative with volume
        elif trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite TRIX crossover
        elif trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0 and position == 1:
            position = 0
            signals[i] = 0.0
        elif trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0 and position == -1:
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