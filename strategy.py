#!/usr/bin/env python3
# 4h Ichimoku Cloud Breakout + Volume Spike + Daily EMA Trend
# Hypothesis: Ichimoku Cloud provides strong support/resistance; breakouts above/below cloud with volume confirmation and daily EMA trend filter capture sustained moves in both bull and bear markets. Designed for low trade frequency (~20-40/year) with clear entry/exit rules.

name = "4h_Ichimoku_Volume_DailyTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Ichimoku Cloud (9, 26, 52) ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for entry)
    
    # The "cloud" is between Senkou Span A and Senkou Span B
    # For simplicity, we use the current cloud boundaries (already shifted in data)
    # We need to align Senkou Span A and B to current periods (they are forward-shifted in calculation)
    # So we shift them back by 26 to align with current price for cloud comparison
    senkou_a_aligned = np.roll(senkou_a, 26)
    senkou_b_aligned = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_aligned[:26] = np.nan
    senkou_b_aligned[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready (52 for Senkou B + buffer)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above cloud + volume spike + price above daily EMA50
            if (close[i] > cloud_top[i] and 
                vol_spike[i] and
                close[i] > ema_50_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud + volume spike + price below daily EMA50
            elif (close[i] < cloud_bottom[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price re-enters cloud (below cloud top)
            if close[i] < cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters cloud (above cloud bottom)
            if close[i] > cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals