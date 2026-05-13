#!/usr/bin/env python3
"""
4h_Pivot_Trend_Follow_HTF_Volume
Hypothesis: Daily pivot points (PP, S1, R1) provide reliable support/resistance. 
Breakout above R1 with daily/weekly uptrend and volume spike = long.
Breakdown below S1 with daily/weekly downtrend and volume spike = short.
Exit when price returns to pivot point or trend reverses.
Uses 4h price for entry/exit, 1d/1w for trend filter, volume confirmation to filter noise.
Target: 20-40 trades/year per symbol.
"""

name = "4h_Pivot_Trend_Follow_HTF_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Pivot Points (calculated from previous day)
    # We need to get daily OHLC to calculate pivots, then align to 4h
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: PP = (H + L + C)/3, S1 = 2*PP - H, R1 = 2*PP - L
    # Use previous day's values (shifted by 1) to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Shift by 1 to use previous day's data for today's pivot
    prev_daily_high = np.concatenate([[np.nan], daily_high[:-1]])
    prev_daily_low = np.concatenate([[np.nan], daily_low[:-1]])
    prev_daily_close = np.concatenate([[np.nan], daily_close[:-1]])
    
    # Calculate pivot points
    pp = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    s1 = 2 * pp - prev_daily_high
    r1 = 2 * pp - prev_daily_low
    
    # Align pivot levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # Weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if pivot data not available (first day)
        if np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Get values
        pp_val = pp_aligned[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 4h uptrend, weekly uptrend filter, volume confirmation
            if close[i] > r1_val and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, 4h downtrend, weekly downtrend filter, volume confirmation
            elif close[i] < s1_val and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: return to pivot point or 4h trend turns down
            if close[i] <= pp_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: return to pivot point or 4h trend turns up
            if close[i] >= pp_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals