#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with weekly trend filter and volume confirmation.
Long when price breaks above Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND weekly close > weekly Kumo top (bullish weekly trend) AND volume > 1.5x 20-period average.
Short when price breaks below Kumo AND Tenkan < Kijun (bearish TK cross) AND weekly close < weekly Kumo bottom (bearish weekly trend) AND volume > 1.5x 20-period average.
Exit when price re-enters Kumo or ATR trailing stop (2.5*ATR from extreme).
Ichimoku provides dynamic support/resistance via Kumo and momentum via TK cross.
Weekly trend filter ensures alignment with higher-timeframe bias, reducing whipsaws.
Volume confirmation ensures institutional participation. Works in bull markets (bullish TK cross + price above cloud in uptrend)
and bear markets (bearish TK cross + price below cloud in downtrend) by following weekly trend.
Target trade frequency: 12-30 trades/year per symbol (48-120 total over 4 years) to avoid fee drag.
"""

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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Kumo (Cloud) top and bottom
    kumomax = np.maximum(senkou_a, senkou_b)
    kumomin = np.minimum(senkou_a, senkou_b)
    
    # Calculate weekly Ichimoku for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Tenkan and Kijun
    period9_high_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (period9_high_1w + period9_low_1w) / 2.0
    
    period26_high_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (period26_high_1w + period26_low_1w) / 2.0
    
    # Weekly Senkou Span A and B
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2.0
    
    period52_high_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (period52_high_1w + period52_low_1w) / 2.0
    
    # Weekly Kumo top and bottom
    kumomax_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    kumomin_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    kumomax_1w_aligned = align_htf_to_ltf(prices, df_1w, kumomax_1w)
    kumomin_1w_aligned = align_htf_to_ltf(prices, df_1w, kumomin_1w)
    
    # Align 6h Ichimoku components (no additional delay needed as they are based on completed periods)
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun)
    kumomax_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kumomax)
    kumomin_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kumomin)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # Ichimoku needs 52, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumomax_aligned[i]) or np.isnan(kumomin_aligned[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(kumomax_1w_aligned[i]) or np.isnan(kumomin_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        kumomax_val = kumomax_aligned[i]
        kumomin_val = kumomin_aligned[i]
        tenkan_1w_val = tenkan_1w_aligned[i]
        kijun_1w_val = kijun_1w_aligned[i]
        kumomax_1w_val = kumomax_1w_aligned[i]
        kumomin_1w_val = kumomin_1w_aligned[i]
        
        if position == 0:
            # Long: Price above Kumo AND bullish TK cross AND weekly bullish trend AND volume spike
            if (close[i] > kumomax_val and tenkan_val > kijun_val and 
                close_1w[i // 20] > kumomax_1w_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price below Kumo AND bearish TK cross AND weekly bearish trend AND volume spike
            elif (close[i] < kumomin_val and tenkan_val < kijun_val and 
                  close_1w[i // 20] < kumomin_1w_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price re-enters Kumo
            if position == 1 and close[i] <= kumomax_val:
                exit_signal = True
            elif position == -1 and close[i] >= kumomin_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Kumo_Breakout_WeeklyTrendFilter_VolumeConfirmation_KumoExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0