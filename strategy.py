#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_WeeklyTrend_v1
Hypothesis: 6h Ichimoku TK cross with 1d Kumo filter and 1w trend filter.
- Long when: TK crosses bullish + price above Kumo (1d) + price > 1w EMA50 (uptrend)
- Short when: TK crosses bearish + price below Kumo (1d) + price < 1w EMA50 (downtrend)
Volume confirmation reduces false signals. Discrete sizing (0.25) minimizes fee churn.
Designed for BTC/ETH: Ichimoku adapts to trend, Kumo provides dynamic S/R, weekly trend filters counter-trend whipsaws.
Target: 50-150 total trades over 4 years (12-37/year) by requiring TK cross, Kumo alignment, weekly trend, and volume.
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
    
    # Load 1d data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df_1d['high'].rolling(window=9, min_periods=9).max()
    period9_low = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df_1d['high'].rolling(window=26, min_periods=26).max()
    period26_low = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = df_1d['high'].rolling(window=52, min_periods=52).max()
    period52_low = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    
    # Kumo (cloud) top/bottom: Senkou Span A/B
    kumotop = np.maximum(senkou_span_a, senkou_span_b)
    kumobottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    kumotop_aligned = align_htf_to_ltf(prices, df_1d, kumotop.values)
    kumobottom_aligned = align_htf_to_ltf(prices, df_1d, kumobottom.values)
    
    # TK cross signals: 1 = bullish cross (tenkan > kijun), -1 = bearish cross (tenkan < kijun)
    tk_cross = np.where(tenkan_sen_aligned > kijun_sen_aligned, 1, -1)
    # Detect crossovers: change in tk_cross signal
    tk_cross_signal = np.zeros(n, dtype=int)
    tk_cross_signal[1:] = np.where(tk_cross[1:] != tk_cross[:-1], tk_cross[1:], 0)
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    htf_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Ichimoku, 50 for weekly EMA, 20 for volume MA)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(kumotop_aligned[i]) or np.isnan(kumobottom_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Bullish TK cross (tenkan crosses above kijun)
        bullish_cross = (tk_cross_signal[i] == 1)
        # Bearish TK cross (tenkan crosses below kijun)
        bearish_cross = (tk_cross_signal[i] == -1)
        
        # Price relative to Kumo
        price_above_kumo = close[i] > kumotop_aligned[i]
        price_below_kumo = close[i] < kumobottom_aligned[i]
        
        if htf_trend[i] == 1:  # Uptrend on 1w
            # Long signal: bullish TK cross above Kumo with volume spike
            if bullish_cross and price_above_kumo and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: bearish TK cross OR price falls below Kumo bottom
            elif bearish_cross or price_below_kumo:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1w
            # Short signal: bearish TK cross below Kumo with volume spike
            if bearish_cross and price_below_kumo and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: bullish TK cross OR price rises above Kumo top
            elif bullish_cross or price_above_kumo:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0