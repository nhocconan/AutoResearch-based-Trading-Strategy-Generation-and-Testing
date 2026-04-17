# 12h_WeeklyEMA34_PivotBreakout_Volume
# Hypothesis: Weekly EMA(34) defines long-term trend, weekly pivot R1/S1 provide breakout levels,
# volume confirms breakout strength, and volatility filter avoids low-momentum periods.
# Works in bull (trend-following breakouts) and bear (mean reversion at pivots with volume).
# Target: 50-150 trades over 4 years (~12-37/year) to stay under fee drag threshold.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly standard deviation for volatility filter
    std_1w = pd.Series(close_1w).rolling(window=34, min_periods=34).std().values
    std_1w_aligned = align_htf_to_ltf(prices, df_1w, std_1w)
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Use previous week's pivots (avoid look-ahead)
    r1_1w_prev = np.roll(r1_1w, 1)
    s1_1w_prev = np.roll(s1_1w, 1)
    r2_1w_prev = np.roll(r2_1w, 1)
    s2_1w_prev = np.roll(s2_1w, 1)
    r1_1w_prev[0] = np.nan
    s1_1w_prev[0] = np.nan
    r2_1w_prev[0] = np.nan
    s2_1w_prev[0] = np.nan
    
    # Align weekly pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1w, r1_1w_prev)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1_1w_prev)
    r2_12h = align_htf_to_ltf(prices, df_1w, r2_1w_prev)
    s2_12h = align_htf_to_ltf(prices, df_1w, s2_1w_prev)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need EMA34, std, pivots, volume MA20, ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or
            np.isnan(s2_12h[i]) or
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(std_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        # Weekly trend filter: price above/below weekly EMA34
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        # Weekly volatility filter: current weekly volatility > average
        vol_filter = std_1w_aligned[i] > 0  # Only trade when volatility exists
        
        if position == 0:
            # Long: price breaks above R1 with volume, volatility AND weekly uptrend
            if (close[i] > r1_12h[i] and volume_filter and volatility_filter and 
                trend_up and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, volatility AND weekly downtrend
            elif (close[i] < s1_12h[i] and volume_filter and volatility_filter and 
                  trend_down and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly EMA34 or volatility drops
            if close[i] < ema34_1w_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly EMA34 or volatility drops
            if close[i] > ema34_1w_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA34_PivotBreakout_Volume"
timeframe = "12h"
leverage = 1.0