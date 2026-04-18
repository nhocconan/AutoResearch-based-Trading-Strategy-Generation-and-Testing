#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly pivot point (from weekly high/low/close) combined with weekly EMA trend
# and volume confirmation on 6h timeframe. Pivot points act as dynamic support/resistance.
# EMA trend filter ensures we trade with the weekly trend. Volume confirms breakout strength.
# Designed for 6h timeframe with target 50-150 trades over 4 years (12-37/year) to avoid fee drag.
# Works in both bull and bear markets by following weekly trend and fading at pivot extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and EMA
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (P) = (H + L + C)/3
    # Support 1 (S1) = 2*P - H
    # Resistance 1 (R1) = 2*P - L
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly data to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 6h ATR for entry threshold
    tr_6h_1 = high - low
    tr_6h_2 = np.abs(high - np.roll(close, 1))
    tr_6h_3 = np.abs(low - np.roll(close, 1))
    tr_6h_1[0] = high[0] - low[0]
    tr_6h_2[0] = np.abs(high[0] - close[0])
    tr_6h_3[0] = np.abs(low[0] - close[0])
    tr_6h = np.maximum(tr_6h_1, np.maximum(tr_6h_2, tr_6h_3))
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # need weekly EMA21 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price above weekly EMA21 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_21_1w_aligned[i]
        trend_down = close[i] < ema_21_1w_aligned[i]
        
        if position == 0:
            # Long entry: price crosses above R1 with volume and uptrend
            if (close[i] > r1_1w_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below S1 with volume and downtrend
            elif (close[i] < s1_1w_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below pivot or EMA21
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot or EMA21
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_EMA21_Trend_Volume"
timeframe = "6h"
leverage = 1.0