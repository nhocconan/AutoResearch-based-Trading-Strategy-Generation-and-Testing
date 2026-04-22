#!/usr/bin/env python3
"""
Hypothesis: 6-hour ADX + Williams Alligator with 1-week trend filter.
Long when ADX > 25 (trending) + Alligator bullish alignment (Lips > Teeth > Jaw) + weekly close > weekly EMA50.
Short when ADX > 25 + Alligator bearish alignment (Lips < Teeth < Jaw) + weekly close < weekly EMA50.
Exit when ADX < 20 (range) or Alligator alignment breaks.
Weekly filter ensures we only trade in strong trends, avoiding whipsaws in ranging markets.
Designed for low trade frequency (~10-20/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(high, 1)), abs(low - np.roll(low, 1))))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    
    jaw_shifted = jaw.shift(8).values
    teeth_shifted = teeth.shift(5).values
    lips_shifted = lips.shift(3).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(weekly_ema50_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_close_price = df_1w['close'].values
        # Get current weekly close value (need to find index)
        # Simplified: use the weekly EMA50 as trend proxy
        weekly_trend_up = weekly_close_price[-1] > weekly_ema50[-1] if len(weekly_close_price) > 0 else False
        weekly_trend_down = weekly_close_price[-1] < weekly_ema50[-1] if len(weekly_close_price) > 0 else False
        
        # Better approach: use aligned weekly close
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
        weekly_close_val = weekly_close_aligned[i]
        weekly_ema50_val = weekly_ema50_aligned[i]
        
        if np.isnan(weekly_close_val) or np.isnan(weekly_ema50_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_trend_up = weekly_close_val > weekly_ema50_val
        weekly_trend_down = weekly_close_val < weekly_ema50_val
        
        if position == 0:
            # Long: ADX > 25 (trending) + Alligator bullish + weekly uptrend
            if (adx[i] > 25 and 
                lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and 
                weekly_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending) + Alligator bearish + weekly downtrend
            elif (adx[i] > 25 and 
                  lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and 
                  weekly_trend_down):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX < 20 (range) or Alligator alignment breaks
                if adx[i] < 20 or not (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX < 20 (range) or Alligator alignment breaks
                if adx[i] < 20 or not (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ADX_Alligator_WeeklyTrendFilter"
timeframe = "6h"
leverage = 1.0