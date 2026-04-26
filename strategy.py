#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter_v1
Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1-day trend filter and choppiness regime filter.
In trending markets (price > 1-day EMA34), long R3 breakout or short S3 breakout when market is not choppy (CHOP < 61.8).
R3/S3 represent solid breakout levels, reducing false signals.
Uses 1-day trend filter to ensure we trade with the primary trend.
Choppiness filter avoids whipsaws in ranging markets.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of breakout, trend, and low chop.
Works in bull/bear via 1-day trend filter: only longs in uptrend, shorts in downtrend.
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
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla
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
    
    # Calculate choppiness index on 1d to filter ranging markets
    def calculate_choppiness(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index (CHOP)"""
        atr_sum = np.zeros(len(close_arr))
        true_range = np.maximum(high_arr - low_arr,
                               np.maximum(np.abs(high_arr - np.roll(close_arr, 1)),
                                          np.abs(low_arr - np.roll(close_arr, 1))))
        true_range[0] = high_arr[0] - low_arr[0]  # First TR
        
        # Calculate ATR using Wilder's smoothing (equivalent to RMA)
        atr = np.zeros(len(close_arr))
        atr[period-1] = np.mean(true_range[:period])
        for i in range(period, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        
        # Calculate highest high and lowest low over period
        highest_high = np.zeros(len(close_arr))
        lowest_low = np.zeros(len(close_arr))
        for i in range(len(close_arr)):
            if i < period:
                highest_high[i] = np.max(high_arr[:i+1])
                lowest_low[i] = np.min(low_arr[:i+1])
            else:
                highest_high[i] = np.max(high_arr[i-period+1:i+1])
                lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        
        # CHOP = 100 * log10(sum(ATR)/ (HHH - LLL)) / log10(period)
        chop = np.full(len(close_arr), np.nan)
        for i in range(period-1, len(close_arr)):
            atr_sum_period = np.sum(atr[i-period+1:i+1])
            hh_ll = highest_high[i] - lowest_low[i]
            if hh_ll > 0:
                chop[i] = 100 * np.log10(atr_sum_period / hh_ll) / np.log10(period)
        return chop
    
    chop_1d = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 14 for CHOP)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Chop filter: only trade when market is not too choppy (CHOP < 61.8 = trending)
        not_choppy = chop_1d_aligned[i] < 61.8
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1 and not_choppy:  # Uptrend on 1d and not choppy
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
        elif htf_trend[i] == -1 and not_choppy:  # Downtrend on 1d and not choppy
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
            # Either no trend or choppy market - hold flat or current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0