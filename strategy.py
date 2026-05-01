#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(9) zero-line cross with 1d volume spike confirmation and chop regime filter
# TRIX is a triple-smoothed EMA momentum oscillator that filters noise and identifies trend changes
# Zero-line cross provides clean entry signals with low lag
# 1d volume spike confirms institutional participation in the breakout
# Choppiness index regime filter ensures we only trade in trending markets (CHOP < 38.2) or mean-revert in range (CHOP > 61.8)
# Designed for low trade frequency (20-40/year) with high edge to overcome fee drag
# Works in both bull and bear markets by adapting to regime

name = "4h_TRIX_ZeroCross_1dVolumeSpike_ChopRegime_v1"
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
    
    # 1d HTF data for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (vol_ma_20 * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1d Choppiness Index: CHOP > 61.8 = range, CHOP < 38.2 = trend
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_1d = np.zeros(len(df_1d))
    tr_1d = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1))
    tr_1d = np.maximum(tr_1d, np.roll(df_1d['low'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr_1d = np.maximum(tr_1d, np.roll(df_1d['high'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['high'].values, 1))
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first TR
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_1d = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)  # fill NaN with neutral
    
    chop_regime_trending = chop_1d < 38.2   # trending regime
    chop_regime_ranging = chop_1d > 61.8    # ranging regime
    chop_regime_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_trending)
    chop_regime_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_ranging)
    
    # TRIX(9) on 4h close: triple EMA, then percent change
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = pd.Series(ema3).pct_change(periods=1) * 100  # percent change
    trix = np.nan_to_num(trix, nan=0.0)
    
    # TRIX zero-line cross signals
    trix_cross_up = (trix > 0) & (np.roll(trix, 1) <= 0)   # crossed above zero
    trix_cross_down = (trix < 0) & (np.roll(trix, 1) >= 0) # crossed below zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(30, 20)  # TRIX needs ~30 bars for stability, plus 1d indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_regime_trending_aligned[i]) or np.isnan(chop_regime_ranging_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike_1d_aligned[i]
        chop_trend = chop_regime_trending_aligned[i]
        chop_range = chop_regime_ranging_aligned[i]
        trix_up = trix_cross_up[i]
        trix_down = trix_cross_down[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: TRIX crosses above zero, volume spike, trending regime
            if trix_up and vol_spike and chop_trend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, volume spike, trending regime
            elif trix_down and vol_spike and chop_trend:
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging regime: TRIX extremes
            elif chop_range:
                if trix < -2.0 and vol_spike:  # oversold, mean reversion long
                    signals[i] = 0.20
                    position = 1
                elif trix > 2.0 and vol_spike:  # overbought, mean reversion short
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on TRIX cross below zero or loss of momentum
            if trix_down or (chop_range and trix > 0.5):  # exit ranging long on mean reversion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on TRIX cross above zero or loss of momentum
            if trix_up or (chop_range and trix < -0.5):  # exit ranging short on mean reversion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals