#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v1
# Strategy: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide strong intraday support/resistance. 
# Breakouts above resistance or below support with 1d EMA trend alignment and volume confirmation 
# capture institutional moves. Works in bull markets via long breakouts and bear markets via short breakdowns.
# Designed for low trade frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    
    # Calculate Camarilla levels for previous day
    close_prev = df_1d['close'].shift(1)
    high_prev = df_1d['high'].shift(1)
    low_prev = df_1d['low'].shift(1)
    
    typical_price_prev = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels (based on previous day)
    S1 = typical_price_prev - (1.1 * range_prev / 12)
    S2 = typical_price_prev - (1.1 * range_prev / 6)
    S3 = typical_price_prev - (1.1 * range_prev / 4)
    R1 = typical_price_prev + (1.1 * range_prev / 12)
    R2 = typical_price_prev + (1.1 * range_prev / 6)
    R3 = typical_price_prev + (1.1 * range_prev / 4)
    
    # Align Camarilla levels to 4h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2.values)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Price levels
        R1_level = R1_aligned[i]
        R2_level = R2_aligned[i]
        R3_level = R3_aligned[i]
        S1_level = S1_aligned[i]
        S2_level = S2_aligned[i]
        S3_level = S3_aligned[i]
        
        # 1d EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above R3 AND bullish trend AND volume confirmation
        if close[i] > R3_level and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below S3 AND bearish trend AND volume confirmation
        elif close[i] < S3_level and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price returns to opposite S1/R1 level (mean reversion to mean)
        elif position == 1 and close[i] < S1_level:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > R1_level:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals