#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_chop_v1
# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter.
# Long: Price touches or breaks above H3 pivot level with volume > 1.5x 20-period average and CHOP > 50 (range/mean-reversion regime).
# Short: Price touches or breaks below L3 pivot level with volume > 1.5x 20-period average and CHOP > 50.
# Exit: Price returns to opposite pivot level (long exits below H3, short exits above L3) or CHOP < 30 (strong trend regime).
# Uses 1d trend filter: only long when 1d close > 1d EMA50, only short when 1d close < 1d EMA50.
# Target: 20-40 trades/year to minimize fee drag while maintaining edge.
# Camarilla pivots provide precise intraday support/resistance levels that work in ranging markets.
# Volume confirmation ensures institutional participation. CHOP filter avoids strong trending regimes where mean reversion fails.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_chop_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # H2 = close + 0.55 * (high - low)
    # H1 = close + 0.275 * (high - low)
    # L1 = close - 0.275 * (high - low)
    # L2 = close - 0.55 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    daily_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * daily_range
    camarilla_l3 = close_1d - 1.1 * daily_range
    
    # Align 1d data to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    # Simplified version: CHOP = 100 * log10( sum(tr_true_range) / log10(14) / (max_high - min_low) ) over 14 periods
    # We'll use a common approximation: CHOP = 100 * log10( sum(ATR) / log10(n) / (HHV - LLV) )
    # For practical purposes, we'll use: CHOP > 50 indicates ranging market, CHOP < 30 indicates trending
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range calculation
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14)
    atr_14 = true_range.rolling(window=14, min_periods=14).mean()
    
    # Sum of ATR over 14 periods
    sum_atr_14 = atr_14.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = high_s.rolling(window=14, min_periods=14).max()
    lowest_low_14 = low_s.rolling(window=14, min_periods=14).min()
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr_14 / (np.log10(14) * (highest_high_14 - lowest_low_14)))
    chop_values = chop.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(volume[i]) or np.isnan(close[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # 1d trend filter: close > EMA50 for uptrend, < EMA50 for downtrend
        trend_1d_up = close[i] > ema_50_1d_aligned[i]  # Using 4h close vs 1d EMA (aligned)
        trend_1d_down = close[i] < ema_50_1d_aligned[i]
        # Choppiness regime filter: CHOP > 50 indicates ranging/mean-reversion regime
        chop_regime = chop_values[i] > 50
        # Strong trend regime filter: CHOP < 30 indicates strong trend (avoid mean reversion)
        strong_trend_regime = chop_values[i] < 30
        
        if position == 1:  # Long position
            # Exit: Price returns to Camarilla H3 level or enters strong trend regime
            if close[i] <= camarilla_h3_aligned[i] or strong_trend_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Camarilla L3 level or enters strong trend regime
            if close[i] >= camarilla_l3_aligned[i] or strong_trend_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price touches/breaks above H3 with volume, uptrend, and ranging regime
            if (close[i] >= camarilla_h3_aligned[i] and    # Touch/break above H3
                volume_confirmed and                       # Volume spike
                trend_1d_up and                            # 1d uptrend
                chop_regime):                              # Ranging market (mean reversion favorable)
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches/breaks below L3 with volume, downtrend, and ranging regime
            elif (close[i] <= camarilla_l3_aligned[i] and  # Touch/break below L3
                  volume_confirmed and                     # Volume spike
                  trend_1d_down and                        # 1d downtrend
                  chop_regime):                            # Ranging market (mean reversion favorable)
                position = -1
                signals[i] = -0.25
    
    return signals