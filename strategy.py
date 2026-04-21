#!/usr/bin/env python3
"""
4h_HTF_1d_TRIX_VolumeSpike_ChopRegime_V1
Hypothesis: 4h TRIX (triple EMA) momentum with 1d volume spike confirmation and 1d choppiness regime filter. 
TRIX > 0 indicates bullish momentum, TRIX < 0 bearish. Volume spike (>1.5x 20-period MA) confirms conviction. 
Choppiness regime (CHOP > 61.8 = range, < 38.2 = trend) ensures we only trade in trending markets on 1d. 
In trending regime (CHOP < 38.2): long when TRIX crosses above zero, short when crosses below zero. 
In ranging regime (CHOP > 61.8): fade extremes - long when TRIX < -0.05, short when TRIX > 0.05. 
This adapts to both bull/bear markets by switching between trend-following and mean-reversion based on 1d chop.
Target 20-50 trades/year (80-200 total over 4 years). Uses 4h primary timeframe with 1d HTF for volume and chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for volume and choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Volume MA for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1d Choppiness Index (CHOP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Add first TR (for period 0) as high-low
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    # ATR (14-period)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(tr_sum / (hh_1d - ll_1d)) / log10(14)
    # Avoid division by zero
    range_hl = hh_1d - ll_1d
    chop_1d = np.where(range_hl > 0, 100 * np.log10(tr_sum / range_hl) / np.log10(14), 50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # TRIX: triple EMA of close, then ROC
    # EMA1
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (EMA3_today - EMA3_yesterday) / EMA3_yesterday
    trix = np.zeros_like(close_4h)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    # Avoid division by zero in first value
    trix[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(trix[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma_1d_aligned[i]  # volume confirmation
        chop = chop_1d_aligned[i]
        trix_val = trix[i]
        
        if position == 0:
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            if chop < 38.2:  # Trending regime
                # Long: TRIX crosses above zero
                if trix_val > 0 and trix[i-1] <= 0 and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX crosses below zero
                elif trix_val < 0 and trix[i-1] >= 0 and vol_ok:
                    signals[i] = -0.25
                    position = -1
            elif chop > 61.8:  # Ranging regime
                # Long: TRIX deeply negative (oversold)
                if trix_val < -0.05 and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX deeply positive (overbought)
                elif trix_val > 0.05 and vol_ok:
                    signals[i] = -0.25
                    position = -1
            # Else: choppy regime (38.2 <= CHOP <= 61.8) - no trades
        
        elif position == 1:
            # Exit long: regime change or TRIX signal reversal
            if chop > 61.8:  # Entered ranging regime
                signals[i] = 0.0
                position = 0
            elif trix_val < 0:  # TRIX turned bearish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: regime change or TRIX signal reversal
            if chop > 61.8:  # Entered ranging regime
                signals[i] = 0.0
                position = 0
            elif trix_val > 0:  # TRIX turned bullish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_TRIX_VolumeSpike_ChopRegime_V1"
timeframe = "4h"
leverage = 1.0