#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: On 6h timeframe, use Ichimoku Tenkan-Kijun cross for entry timing,
filtered by price position relative to Kumo (cloud) from 1d timeframe and 1d trend.
Tenkan-Kijun cross provides timely entries, while 1d cloud acts as dynamic support/resistance
and 1d EMA50 filter ensures we trade with higher timeframe trend.
This combination should work in both bull and bear markets by avoiding counter-trend trades
and using cloud as volatility-adjusted target zone.
Target: 12-30 trades/year on 6h timeframe (~50-120 over 4 years).
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
    
    # Calculate Ichimoku components on 6h data
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
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # 1d HTF data for cloud and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku cloud components
    # Tenkan-sen 1d
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    # Kijun-sen 1d
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    # Senkou Span A 1d
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B 1d
    high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_senkou_b_1d + low_senkou_b_1d) / 2
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe (completed 1d bar only)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumO_top_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    kumO_bottom_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # TK cross signals (bullish: Tenkan > Kijun, bearish: Tenkan < Kijun)
    tk_bullish = tenkan > kijun
    tk_bearish = tenkan < kijun
    
    # Price above/below cloud
    price_above_kumo = close > kumO_top_1d
    price_below_kumo = close < kumO_bottom_1d
    
    # 1d trend filter: price relative to EMA50
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52 periods) and 1d EMA50
    start_idx = max(52, 50)  # Ichimoku needs 52 for Senkou B, EMA50 needs 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumO_top_1d[i]) or np.isnan(kumO_bottom_1d[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish TK cross + price above Kumo + 1d uptrend
            long_setup = tk_bullish[i] and price_above_kumo[i] and uptrend_1d[i]
            # Short: bearish TK cross + price below Kumo + 1d downtrend
            short_setup = tk_bearish[i] and price_below_kumo[i] and downtrend_1d[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish TK cross OR price falls below Kumo bottom OR 1d trend turns down
            if tk_bearish[i] or (close[i] < kumO_bottom_1d[i]) or (not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish TK cross OR price rises above Kumo top OR 1d trend turns up
            if tk_bullish[i] or (close[i] > kumO_top_1d[i]) or (not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0