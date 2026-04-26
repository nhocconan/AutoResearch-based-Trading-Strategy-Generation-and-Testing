#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter_v1
Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1-day trend filter and choppiness regime filter.
Only trade breakouts in the direction of the 1-day EMA50 trend when market is not choppy (Choppiness Index < 61.8).
R3/S3 levels provide reliable breakout points with lower false signals than R4/S4 in sideways markets.
Uses 1-day trend filter to ensure alignment with primary trend and chop filter to avoid whipsaws in ranging markets.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of breakout, trend, and low-chop regime.
Works in bull/bear via 1-day trend filter: only takes long breakouts in uptrend, short in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Choppiness Index on 1d for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(n)
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first TR
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned < 61.8  # true when not choppy (trending market)
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R3_1d = typical_price_1d + (1.1/4) * (df_1d['high'] - df_1d['low'])  # R3 level
    S3_1d = typical_price_1d - (1.1/4) * (df_1d['high'] - df_1d['low'])  # S3 level
    
    # Align Camarilla levels to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d.values)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 14 for chop)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Only trade when market is not choppy (trending regime)
        if chop_filter[i]:
            # Breakout conditions with trend filter
            if htf_trend[i] == 1:  # Uptrend on 1d
                # Long breakout above R3
                if close[i] > R3_1d_aligned[i]:
                    if position != 1:
                        signals[i] = 0.25
                        position = 1
                    else:
                        signals[i] = 0.25
                # Exit long if price falls below S3 (reversal signal)
                elif position == 1 and close[i] < S3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            elif htf_trend[i] == -1:  # Downtrend on 1d
                # Short breakdown below S3
                if close[i] < S3_1d_aligned[i]:
                    if position != -1:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = -0.25
                # Exit short if price rises above R3 (reversal signal)
                elif position == -1 and close[i] > R3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                # Should not happen with our trend calculation
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Choppy market - exit any position and stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0