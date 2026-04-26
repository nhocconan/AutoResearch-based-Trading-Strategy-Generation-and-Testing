#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_ChopFilter_v2
Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1-week trend filter and choppiness regime filter.
Only trade breakouts in the direction of the 1-week EMA50 trend when the market is not too choppy (CHOP < 61.8).
R3/S3 levels provide strong breakout signals with fewer false breaks than R1/S1.
Uses 1-week trend filter to ensure we only trade with the primary trend, working in both bull and bear markets.
Chop filter prevents whipsaws in ranging markets. Designed for 12-37 trades/year (50-150 over 4 years).
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
    
    # Load 1w data ONCE before loop for HTF trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for HTF trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    htf_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R3_1d = typical_price_1d + 1.1 * (df_1d['high'] - df_1d['low']) / 4  # R3 level
    S3_1d = typical_price_1d - 1.1 * (df_1d['high'] - df_1d['low']) / 4  # S3 level
    
    # Align Camarilla levels to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d.values)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d.values)
    
    # Calculate choppiness index on 1d data (need high, low, close)
    def choppiness_index(high_arr, low_arr, close_arr, window=14):
        """Calculate choppiness index: higher = more choppy, lower = more trending"""
        atr_sum = np.zeros_like(close_arr)
        true_range = np.zeros_like(close_arr)
        
        # Calculate true range
        true_range[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = abs(high_arr[i] - close_arr[i-1])
            lc = abs(low_arr[i] - close_arr[i-1])
            true_range[i] = max(hl, hc, lc)
        
        # Calculate ATR sum
        for i in range(window-1, len(close_arr)):
            atr_sum[i] = np.sum(true_range[i-window+1:i+1])
        
        # Calculate choppiness
        chop = np.full_like(close_arr, 50.0)  # default neutral
        for i in range(window-1, len(close_arr)):
            if atr_sum[i] > 0:
                max_high = np.max(high_arr[i-window+1:i+1])
                min_low = np.min(low_arr[i-window+1:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(atr_sum[i] / (max_high - min_low)) / np.log10(window)
        return chop
    
    chop_1d = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Chop regime: < 61.8 = trending (good for breakouts), > 61.8 = choppy (avoid)
    chop_regime = chop_1d_aligned < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA, 14 for chop)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Only trade in trending regimes (chop < 61.8)
        if not chop_regime[i]:
            # In choppy market, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 1w
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
        elif htf_trend[i] == -1:  # Downtrend on 1w
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

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_ChopFilter_v2"
timeframe = "12h"
leverage = 1.0