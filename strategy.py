#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter
# - Use Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) on 6h
# - Long when price > Kumo (cloud) and TK cross bullish, confirmed by 1d uptrend (price > 1d EMA200)
# - Short when price < Kumo and TK cross bearish, confirmed by 1d downtrend (price < 1d EMA200)
# - Exit when TK cross reverses or price enters cloud
# - Ichimoku provides built-in support/resistance and trend strength
# - 1d EMA200 filter ensures we trade with higher timeframe trend
# - Target: 50-150 trades over 4 years

name = "6h_ichimoku_1dema200_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For backtesting, we use current Senkou spans as cloud (no look-ahead)
    # Kumo top = max(Senkou A, Senkou B)
    # Kumo bottom = min(Senkou A, Senkou B)
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK Cross: Tenkan crossing above/below Kijun
    tk_cross = tenkan - kijun
    tk_cross_above = (tk_cross > 0) & (np.roll(tk_cross, 1) <= 0)  # bullish cross
    tk_cross_below = (tk_cross < 0) & (np.roll(tk_cross, 1) >= 0)  # bearish cross
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Wait for Senkou B to stabilize
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(ema_200_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: TK cross bearish OR price enters cloud (below Kumo top)
            if tk_cross_below[i] or close[i] < kumo_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross bullish OR price enters cloud (above Kumo bottom)
            if tk_cross_above[i] or close[i] > kumo_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price outside cloud + TK cross + 1d trend filter
            price_above_kumo = close[i] > kumo_top[i]
            price_below_kumo = close[i] < kumo_bottom[i]
            bullish_tk = tk_cross[i] > 0
            bearish_tk = tk_cross[i] < 0
            uptrend_1d = close[i] > ema_200_aligned[i]
            downtrend_1d = close[i] < ema_200_aligned[i]
            
            # Long: price above cloud + bullish TK cross + 1d uptrend
            if price_above_kumo and bullish_tk and uptrend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + bearish TK cross + 1d downtrend
            elif price_below_kumo and bearish_tk and downtrend_1d:
                signals[i] = -0.25
                position = -1
    
    return signals