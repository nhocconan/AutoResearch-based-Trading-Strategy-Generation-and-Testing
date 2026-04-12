#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_choppiness_breakout_v1
# Uses daily volatility regime (Choppiness Index) to filter breakouts on 12h chart.
# In trending regimes (CHOP < 38.2): take Donchian(20) breakouts with volume confirmation.
# In ranging regimes (CHOP > 61.8): mean revert at daily pivot with tighter stops.
# Designed for low trade frequency (target: 15-25 trades/year) to avoid fee drag.
# Works in bull markets via trend-following breakouts and in bear/ranging via mean reversion.

name = "12h_1d_choppiness_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index and pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum(ATR)/ (max(high)-min(low))) / log10(14)
    sum_atr = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        sum_atr[i] = np.nansum(atr[i-13:i+1])
    
    max_high = np.full(len(high_1d), np.nan)
    min_low = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        max_high[i] = np.nanmax(high_1d[i-13:i+1])
        min_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Daily pivot point (based on previous day)
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Align daily data to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # 12h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if np.isnan(chop_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(high_ma[i]) or np.isnan(low_ma[i]):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        
        # Regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if is_trending:
            # Trend-following: Donchian breakout with volume confirmation
            if high[i] > high_ma[i] and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif low[i] < low_ma[i] and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        elif is_ranging:
            # Mean reversion: revert to daily pivot
            if close[i] > pp_aligned[i] and position != -1:
                # Short when above pivot (expect reversion down)
                position = -1
                signals[i] = -0.20
            elif close[i] < pp_aligned[i] and position != 1:
                # Long when below pivot (expect reversion up)
                position = 1
                signals[i] = 0.20
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.20
                elif position == -1:
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
        else:
            # Choppy middle zone: stay flat
            signals[i] = 0.0
            position = 0
    
    return signals