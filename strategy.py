#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_Breakout_TrendFilter_v1
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 and 1d EMA200 trend filters.
Only trade in direction of higher timeframe trends to avoid counter-trend whipsaws.
Requires volume spike (>1.5x 20-period average) for confirmation.
Uses 0.20 position size to limit risk and reduce fee churn.
Designed for 15-30 trades/year (60-120 over 4 years) by requiring confluence of:
1. 1h price break above R3 or below S3
2. 4h EMA50 trend alignment (price > EMA50 for longs, < EMA50 for shorts)
3. 1d EMA200 trend alignment (price > EMA200 for longs, < EMA200 for shorts)
4. Volume spike confirmation
Works in bull/bear via dual trend filters: only takes longs when both HTFs are bullish,
shorts when both are bearish. Flat when trends disagree or are weak.
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
    
    # Load 4h data ONCE before loop for HTF trend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for HTF trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for HTF trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for HTF trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R3_1d = typical_price_1d + (1.1/4) * (df_1d['high'] - df_1d['low'])  # R3 level
    S3_1d = typical_price_1d - (1.1/4) * (df_1d['high'] - df_1d['low'])  # S3 level
    
    # Align Camarilla levels to 1h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d.values)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 200 for 1d EMA, 50 for 4h EMA, 20 for volume MA)
    start_idx = max(200, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine trend alignment
        # 1 = uptrend (price > both EMAs), -1 = downtrend (price < both EMAs), 0 = mixed/unclear
        if close[i] > ema_50_4h_aligned[i] and close[i] > ema_200_1d_aligned[i]:
            trend_alignment = 1  # bullish
        elif close[i] < ema_50_4h_aligned[i] and close[i] < ema_200_1d_aligned[i]:
            trend_alignment = -1  # bearish
        else:
            trend_alignment = 0  # mixed or transitional
        
        # Entry logic
        if trend_alignment == 1:  # Bullish alignment
            # Long breakout above R3 with volume spike
            if close[i] > R3_1d_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.20
            # Exit long if price falls below S3 (reversal signal)
            elif position == 1 and close[i] < S3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
                    
        elif trend_alignment == -1:  # Bearish alignment
            # Short breakdown below S3 with volume spike
            if close[i] < S3_1d_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = -0.20
            # Exit short if price rises above R3 (reversal signal)
            elif position == -1 and close[i] > R3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
                    
        else:  # Mixed trend - no new entries, only exit existing positions
            if position == 1 and close[i] < S3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > R3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Camarilla_Breakout_TrendFilter_v1"
timeframe = "1h"
leverage = 1.0