#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Ichimoku cloud breakouts on 6h with 1-day trend filter (price >/ < Kumo twist) and volume confirmation capture high-probability momentum in both bull and bear markets. The 1-day EMA50/200 cross provides a stable regime filter, while volume ensures breakout conviction. Targeting 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 and EMA200 for trend regime filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Ichimoku components on 6h (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align Senkou Span A/B to current 6h (they are plotted 26 periods ahead)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=0)  # already forward-shifted in calc
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=0)
    
    # Volume spike detection on 6h (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(period_senkou_b, period_kijun, 50, 200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend regime filter (EMA50 > EMA200 = bull, EMA50 < EMA200 = bear)
        bull_regime = ema_50_aligned[i] > ema_200_aligned[i]
        bear_regime = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Kumo (cloud) boundaries: upper = max(Senkou A, Senkou B), lower = min(Senkou A, Senkou B)
        kumo_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        kumo_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Price above/below cloud
        price_above_kumo = close[i] > kumo_top
        price_below_kumo = close[i] < kumo_bottom
        
        # TK Cross (Tenkan-sen crosses Kijun-sen)
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Long logic: price breaks above cloud with TK cross up + volume spike + bull regime
        if price_above_kumo and tk_cross_up and volume_spike[i] and bull_regime:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below cloud with TK cross down + volume spike + bear regime
        elif price_below_kumo and tk_cross_down and volume_spike[i] and bear_regime:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite cloud boundary or TK cross reverses
        elif position == 1 and (price_below_kumo or tk_cross_down):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (price_above_kumo or tk_cross_up):
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
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0