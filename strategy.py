#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(9) signal line crossover with 1d volume spike and choppiness regime filter
# Long when TRIX crosses above its signal line + 1d volume > 2x 20-period average + CHOP > 61.8 (range)
# Short when TRIX crosses below its signal line + 1d volume > 2x 20-period average + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
# TRIX is a momentum oscillator that filters noise; signal line crossovers catch reversals.
# Volume spike confirms participation; chop filter ensures we only trade in ranging markets where mean reversion works.
# Works in bull markets (buy dips in range) and bear markets (sell rallies in range) by requiring range regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: TRIX, Signal Line, Volume SMA, Choppiness Index ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # TRIX: triple EMA of ROC, period=9
    roc = np.diff(np.log(close_1d), prepend=np.log(close_1d[0]))  # approx ROC
    ema1 = pd.Series(roc).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)  # % change
    trix[0] = 0
    
    # Signal line: EMA of TRIX, period=9
    signal_line = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume SMA (20-period)
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP), period=14
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop = np.where((max_high - min_low) > 0, chop, 50)  # avoid division by zero
    
    # Align all 1d indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    signal_line_aligned = align_htf_to_ltf(prices, df_1d, signal_line)
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators: None needed beyond price and volume ===
    # Volume SMA (20-period) for 4h volume confirmation
    vol_sma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20)  # TRIX needs ~30 bars, volume SMA needs 20
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_4h[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(signal_line_aligned[i]) or
            np.isnan(vol_sma_20_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_sma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. TRIX crosses above signal line
        # 2. 1d volume > 2x 20-period average (confirmation)
        # 3. Choppiness Index > 61.8 (range regime)
        if (trix_aligned[i] > signal_line_aligned[i]) and \
           (trix_aligned[i-1] <= signal_line_aligned[i-1]) and \
           vol_confirm and (chop_aligned[i] > 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. TRIX crosses below signal line
        # 2. 1d volume > 2x 20-period average (confirmation)
        # 3. Choppiness Index > 61.8 (range regime)
        elif (trix_aligned[i] < signal_line_aligned[i]) and \
             (trix_aligned[i-1] >= signal_line_aligned[i-1]) and \
             vol_confirm and (chop_aligned[i] > 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_TRIX9_VolumeSpike_CHOP_Filter_v1"
timeframe = "4h"
leverage = 1.0