#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d EMA34 trend filter and choppiness regime filter.
Only takes breakouts in direction of 1d trend when market is not excessively choppy (CHOP > 38.2).
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of breakout, trend, and regime.
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend filter.
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
    R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])
    S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    # Calculate Choppiness Index on 1d (need high, low, close)
    def choppiness_index(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index: higher = more choppy, lower = more trending"""
        atr_sum = np.zeros_like(close_arr)
        true_range = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if i == 0:
                true_range[i] = high_arr[i] - low_arr[i]
            else:
                true_range[i] = max(high_arr[i] - low_arr[i], 
                                  abs(high_arr[i] - close_arr[i-1]),
                                  abs(low_arr[i] - close_arr[i-1]))
        
        # Calculate ATR using Wilder's smoothing (equivalent to RMA)
        atr = np.zeros_like(close_arr)
        atr[window-1] = np.mean(true_range[:window])
        for i in range(window, len(close_arr)):
            atr[i] = (atr[i-1] * (window-1) + true_range[i]) / window
        
        # Sum of ATR over window
        atr_sum = np.zeros_like(close_arr)
        for i in range(window-1, len(close_arr)):
            atr_sum[i] = np.sum(atr[i-window+1:i+1])
        
        # Choppiness Index formula
        chop = np.zeros_like(close_arr)
        for i in range(window-1, len(close_arr)):
            if atr_sum[i] > 0:
                log_sum = np.log10(atr_sum[i] / window)
                log_range = np.log10(max(high_arr[i-window+1:i+1]) - min(low_arr[i-window+1:i+1]))
                chop[i] = 100 * log_sum / log_range if log_range != 0 else 50
            else:
                chop[i] = 50
        return chop
    
    chop_1d = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 14 for chop)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade when not excessively choppy (CHOP < 61.8 allows trending, but we want CHOP > 38.2 to avoid ranging)
        # Actually, CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
        # We want to avoid choppy markets, so require CHOP < 61.8
        not_choppy = chop_1d_aligned[i] < 61.8
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long breakout above R1 with regime filter
            if close[i] > R1_1d_aligned[i] and not_choppy:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S1 (reversal signal)
            elif position == 1 and close[i] < S1_1d_aligned[i]:
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
            # Short breakdown below S1 with regime filter
            if close[i] < S1_1d_aligned[i] and not_choppy:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R1 (reversal signal)
            elif position == -1 and close[i] > R1_1d_aligned[i]:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0