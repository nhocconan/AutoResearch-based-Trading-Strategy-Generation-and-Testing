#!/usr/bin/env python3
"""
4h_1d_MeanReversion_Stochastic_v1
Hypothesis: Use daily stochastic oscillator (14,3,3) for mean reversion signals in ranging markets.
Filter by 1d ADX < 25 to avoid trending markets. Enter long when %K < 20 and %D < %K (bullish crossover),
enter short when %K > 80 and %D > %K (bearish crossover). Exit when stochastic returns to neutral range (40-60).
Uses 4h timeframe for entries with 1d filter to reduce whipsaws. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by fading extremes only in ranging markets.
Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_MeanReversion_Stochastic_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for stochastic and ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate stochastic oscillator (14,3,3) on daily data
    def stochastic(high, low, close, k_period=14, d_period=3):
        if len(close) < k_period:
            return np.full(len(close), np.nan), np.full(len(close), np.nan)
        
        lowest_low = np.full(len(low), np.nan)
        highest_high = np.full(len(high), np.nan)
        
        for i in range(k_period-1, len(low)):
            lowest_low[i] = np.min(low[i-k_period+1:i+1])
            highest_high[i] = np.max(high[i-k_period+1:i+1])
        
        # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        k_percent = np.full(len(close), np.nan)
        for i in range(k_period-1, len(close)):
            if highest_high[i] != lowest_low[i]:
                k_percent[i] = (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i]) * 100
            else:
                k_percent[i] = 50.0  # Avoid division by zero
        
        # %D = SMA of %K, 3-period
        d_percent = np.full(len(close), np.nan)
        for i in range(k_period-1 + d_period-1, len(k_percent)):
            d_percent[i] = np.mean(k_percent[i-d_period+1:i+1])
        
        return k_percent, d_percent
    
    # Calculate ADX (14) on daily data for trend filter
    def adx(high, low, close, period=14):
        if len(close) < period * 2:
            return np.full(len(close), np.nan)
        
        # True Range
        tr = np.full(len(close), np.nan)
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Directional Movement
        plus_dm = np.full(len(close), np.nan)
        minus_dm = np.full(len(close), np.nan)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed values
        atr = np.full(len(close), np.nan)
        smooth_plus_dm = np.full(len(close), np.nan)
        smooth_minus_dm = np.full(len(close), np.nan)
        
        # Initial averages
        if len(close) >= period:
            atr[period-1] = np.nanmean(tr[1:period+1])
            smooth_plus_dm[period-1] = np.nanmean(plus_dm[1:period+1])
            smooth_minus_dm[period-1] = np.nanmean(minus_dm[1:period+1])
            
            # Wilder's smoothing
            for i in range(period, len(close)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                smooth_plus_dm[i] = (smooth_plus_dm[i-1] * (period-1) + plus_dm[i]) / period
                smooth_minus_dm[i] = (smooth_minus_dm[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(len(close), np.nan)
        minus_di = np.full(len(close), np.nan)
        dx = np.full(len(close), np.nan)
        
        for i in range(period-1, len(close)):
            if atr[i] != 0:
                plus_di[i] = (smooth_plus_dm[i] / atr[i]) * 100
                minus_di[i] = (smooth_minus_dm[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        # ADX = SMA of DX
        adx_vals = np.full(len(close), np.nan)
        for i in range(2*period-2, len(dx)):
            if not np.isnan(dx[i-period+1:i+1]).any():
                adx_vals[i] = np.nanmean(dx[i-period+1:i+1])
        
        return adx_vals
    
    # Calculate indicators on daily data
    stoch_k, stoch_d = stochastic(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        14, 3
    )
    adx_vals = adx(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        14
    )
    
    # Align indicators to 4h timeframe
    stoch_k_aligned = align_htf_to_ltf(prices, df_1d, stoch_k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_1d, stoch_d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_vals)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(stoch_k_aligned[i]) or np.isnan(stoch_d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Range filter: ADX < 25 indicates ranging market
        ranging = adx_aligned[i] < 25
        
        # Stochastic signals
        stoch_k_val = stoch_k_aligned[i]
        stoch_d_val = stoch_d_aligned[i]
        stoch_k_prev = stoch_k_aligned[i-1] if i > 0 else stoch_k_val
        stoch_d_prev = stoch_d_aligned[i-1] if i > 0 else stoch_d_val
        
        # Bullish crossover: %K crosses above %D in oversold territory
        bullish_cross = (stoch_k_prev <= stoch_d_prev and 
                        stoch_k_val > stoch_d_val and 
                        stoch_k_val < 20)
        
        # Bearish crossover: %K crosses below %D in overbought territory
        bearish_cross = (stoch_k_prev >= stoch_d_prev and 
                        stoch_k_val < stoch_d_val and 
                        stoch_k_val > 80)
        
        # Exit when stochastic returns to neutral range (40-60)
        exit_signal = (stoch_k_val >= 40 and stoch_k_val <= 60)
        
        # Entry logic
        long_entry = ranging and bullish_cross
        short_entry = ranging and bearish_cross
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_signal:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_signal:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals