#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1d EMA34 trend filter and choppiness regime.
In trending markets (price > 1d EMA34), long R3 breakout or short S3 breakout.
In choppy markets (Choppiness Index > 61.8), avoid breakout trades to reduce whipsaw.
Uses 1d trend filter to align with daily momentum and avoid counter-trend trades.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of breakout, trend, and regime filter.
Works in bull/bear via 1d trend filter: only takes long breakouts in uptrend, short in downtrend.
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
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R3_1d = typical_price_1d + (1.1/4) * (df_1d['high'] - df_1d['low'])  # R3 level
    S3_1d = typical_price_1d - (1.1/4) * (df_1d['high'] - df_1d['low'])  # S3 level
    
    # Align Camarilla levels to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d.values)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d.values)
    
    # Calculate Choppiness Index on 12h data for regime filter
    def choppiness_index(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First period TR
        
        # Sum of True Range over window
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Choppiness Index formula
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, window=14)
    chop_regime = chop > 61.8  # True = choppy/range, False = trending
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 14 for chop)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Avoid breakout trades in choppy markets
        if chop_regime[i]:
            # In choppy markets, stay flat or reduce position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.0  # Exit long in chop
            else:
                signals[i] = 0.0  # Exit short in chop
            continue
        
        # Breakout conditions with trend filter (only in trending markets)
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
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0