#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_RegimeFilter
# Hypothesis: TRIX momentum on 4h with volume spike and Choppiness regime filter to avoid choppy markets.
# Uses 1d EMA50 for higher timeframe trend filter, TRIX for momentum, volume spike for confirmation.
# Designed for low trade frequency (20-50/year) to minimize fee drift while capturing momentum in trending regimes.
# Works in both bull and bear markets by following the trend direction from higher timeframe.

name = "4h_TRIX_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily trend: EMA50 ---
    prev_close = df_1d['close'].values
    ema_50_1d = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- TRIX on 4h: triple EMA of ROC ---
    # ROC(12)
    roc = np.diff(np.log(close), prepend=np.log(close[0])) * 100
    # EMA1 of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of ROC
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of ROC (TRIX)
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # TRIX is the third EMA of ROC
    
    # --- Choppiness regime filter on 4h ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index: CHOP = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    # Regime: trending if CHOP < 38.2, ranging if CHOP > 61.8
    # We only trade in trending regime (CHOP < 38.2)
    trending_regime = chop < 38.2
    
    # --- Volume confirmation (2.0x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(trix[i]) or
            np.isnan(trending_regime[i]) or
            np.isnan(volume_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from daily EMA50
        bullish_trend = close[i] > ema_50_1d_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: TRIX > 0 (bullish momentum) in bullish trend, trending regime, volume surge
            if trix[i] > 0 and bullish_trend and trending_regime[i] and volume_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX < 0 (bearish momentum) in bearish trend, trending regime, volume surge
            elif trix[i] < 0 and bearish_trend and trending_regime[i] and volume_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: TRIX turns negative OR trend breaks
                if trix[i] < 0 or not bullish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TRIX turns positive OR trend breaks
                if trix[i] > 0 or not bearish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals