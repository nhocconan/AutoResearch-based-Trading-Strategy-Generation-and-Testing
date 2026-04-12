#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_trix_volume_regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for TRIX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate TRIX (15-period triple EMA)
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Get daily data for volume and regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Daily volume ratio (current vs 20-day average)
    vol_ma_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / vol_ma_20d
    
    # Choppiness index regime filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Choppiness Index: 100 * log10(tr_sum / range_max_min) / log10(14)
    chop = np.zeros_like(tr_sum)
    mask = (tr_sum > 0) & (range_max_min > 0)
    chop[mask] = 100 * np.log10(tr_sum[mask] / range_max_min[mask]) / np.log10(14)
    
    # Align all indicators to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix_raw)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1w, trix_signal)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(volume_ratio_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX crossover signals
        trix_cross_above = trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1]
        trix_cross_below = trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1]
        
        # Volume confirmation: above average volume
        volume_ok = volume_ratio_aligned[i] > 1.2
        
        # Regime filter: trending market (Choppiness < 38.2) or ranging (Choppiness > 61.8)
        trending_regime = chop_aligned[i] < 38.2
        ranging_regime = chop_aligned[i] > 61.8
        
        # Entry logic:
        # In trending markets: follow TRIX crossovers
        # In ranging markets: fade extreme TRIX values (mean reversion)
        long_signal = False
        short_signal = False
        
        if trending_regime:
            # Trend following: buy on bullish crossover, sell on bearish crossover
            long_signal = trix_cross_above and volume_ok
            short_signal = trix_cross_below and volume_ok
        elif ranging_regime:
            # Mean reversion: buy when TRIX is deeply oversold, sell when overbought
            long_signal = trix_aligned[i] < -0.5 and volume_ok
            short_signal = trix_aligned[i] > 0.5 and volume_ok
        
        # Exit when signal reverses or volume dries up
        exit_long = (trix_cross_below and trending_regime) or (trix_aligned[i] > 0.2 and ranging_regime)
        exit_short = (trix_cross_above and trending_regime) or (trix_aligned[i] < -0.2 and ranging_regime)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals