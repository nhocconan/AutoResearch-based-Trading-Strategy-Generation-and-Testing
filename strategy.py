#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeRegime_v1
Hypothesis: Daily timeframe strategy using Camarilla R1/S1 breakouts filtered by weekly EMA50 trend and volume regime (choppiness index). Designed for low-frequency, high-conviction entries (target: 7-25 trades/year) that work in both bull and bear markets by requiring alignment with higher timeframe trend and avoiding choppy regimes.
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
    
    # Get weekly data for HTF trend and daily data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close_1d + (1.0/6) * (prev_high_1d - prev_low_1d)
    camarilla_s1 = prev_close_1d - (1.0/6) * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume regime filter using choppiness index (to avoid whipsaw in sideways markets)
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest(high,14) - lowest(low,14))) / log10(14)
    # High chop (>61.8) = ranging market, Low chop (<38.2) = trending market
    # We want to trade when chop < 50 (avoiding extreme chop)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    atr1 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = highest_high14 - lowest_low14
    # Avoid division by zero
    range14 = np.where(range14 == 0, 1e-10, range14)
    chop = 100 * np.log10(sum_atr14 / range14) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # Neutral value when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)  # Already on 1d, but align for safety
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of weekly EMA(50), Camarilla (needs 1d lag), ATR (14), chop (14)
    start_idx = max(50, 1, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        chop_val = chop_aligned[i]
        atr_val = atr[i]
        
        # Trend filter: price > weekly EMA50 (uptrend) or < weekly EMA50 (downtrend)
        uptrend = close_val > ema_50_1w_val
        downtrend = close_val < ema_50_1w_val
        
        # Volume regime filter: avoid extreme chop (choppiness > 61.8) - only trade when chop < 50
        volume_regime_ok = chop_val < 50.0
        
        if position == 0:
            # Long: break above R1 with volume regime ok, and uptrend
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          volume_regime_ok and \
                          uptrend
            
            # Short: break below S1 with volume regime ok, and downtrend
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           volume_regime_ok and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeRegime_v1"
timeframe = "1d"
leverage = 1.0