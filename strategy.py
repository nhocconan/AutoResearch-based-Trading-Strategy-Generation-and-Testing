#!/usr/bin/env python3
"""
12h_Keltner_Channel_Breakout_1wTrend_Volume_v1
Hypothesis: Trade 12-hour breakouts of weekly Keltner Channel (EMA20 + 2*ATR) only when aligned with weekly trend (EMA50) and confirmed by volume spike (>2x average). Uses weekly timeframe for trend direction and 12h for precise entry. Targets 15-30 trades/year with low fee impact. Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation.
"""

name = "12h_Keltner_Channel_Breakout_1wTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get weekly data for Keltner Channel and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate weekly EMA20 for Keltner Channel middle
    ema_20_w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR (14-period) for Keltner Channel width
    tr1 = np.abs(weekly_high - weekly_low)
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: just high-low
    atr_14_w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Keltner Channel bounds
    upper_kc = ema_20_w + (2 * atr_14_w)
    lower_kc = ema_20_w - (2 * atr_14_w)
    
    # Align Keltner Channel levels to 12h timeframe (with 1-bar delay for completed weekly bar)
    upper_kc_aligned = align_htf_to_ltf(prices, df_w, upper_kc, additional_delay_bars=1)
    lower_kc_aligned = align_htf_to_ltf(prices, df_w, lower_kc, additional_delay_bars=1)
    
    # Get weekly trend filter (EMA50)
    ema_50_w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_w_aligned = align_htf_to_ltf(prices, df_w, ema_50_w)
    
    # Get 12h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_kc_aligned[i]) or 
            np.isnan(lower_kc_aligned[i]) or 
            np.isnan(ema_50_w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend using aligned close
        weekly_close_aligned = align_htf_to_ltf(prices, df_w, weekly_close)
        if np.isnan(weekly_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = weekly_close_aligned[i] > ema_50_w_aligned[i]
        trend_down = weekly_close_aligned[i] < ema_50_w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above upper KC with upward trend and volume spike
            if (close[i] > upper_kc_aligned[i] and 
                trend_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below lower KC with downward trend and volume spike
            elif (close[i] < lower_kc_aligned[i] and 
                  trend_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower KC level or trend turns down
            if close[i] < lower_kc_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper KC level or trend turns up
            if close[i] > upper_kc_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals