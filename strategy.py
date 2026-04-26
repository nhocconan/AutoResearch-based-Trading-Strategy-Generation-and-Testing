#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_WeeklyTrend_1dVolatility
Hypothesis: On 6h timeframe, use Ichimoku cloud (TK cross + price vs cloud) filtered by weekly trend (price > weekly EMA50) and 1d volatility expansion (ATR ratio > 1.2).
Enters long when bullish TK cross, price above cloud, weekly uptrend, and expanding volatility.
Enters short when bearish TK cross, price below cloud, weekly downtrend, and expanding volatility.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 total trades over 4 years.
Weekly trend filter ensures alignment with higher timeframe momentum, reducing counter-trend trades in chop.
Volatility filter avoids entries during low-momentum consolidation, improving win rate in both bull and bear markets.
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
    
    # Ichimoku components (9, 26, 52 periods) on 6h
    period9 = 9
    period26 = 26
    period52 = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high9 = pd.Series(high).rolling(window=period9, min_periods=period9).max().values
    low9 = pd.Series(low).rolling(window=period9, min_periods=period9).min().values
    tenkan = (high9 + low9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high26 = pd.Series(high).rolling(window=period26, min_periods=period26).max().values
    low26 = pd.Series(low).rolling(window=period26, min_periods=period26).min().values
    kijun = (high26 + low26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high52 = pd.Series(high).rolling(window=period52, min_periods=period52).max().values
    low52 = pd.Series(low).rolling(window=period52, min_periods=period52).min().values
    senkou_b = ((high52 + low52) / 2.0)
    
    # Current Ichimoku cloud boundaries (Senkou Span A/B from 26 periods ago)
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    
    # Weekly trend filter: price > weekly EMA50 (uptrend) or < weekly EMA50 (downtrend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volatility filter: ATR(14) ratio > 1.2 (expanding volatility)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align to 1d index
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # ATR ratio: current ATR / 50-period ATR average
    atr_ma_50_1d = pd.Series(atr_14_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14_1d / atr_ma_50_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed for TK cross)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # using 1d as reference for alignment structure, but values are 6h
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_lag_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_lag)
    senkou_b_lag_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_lag)
    # Note: tenkan, kijun, senkou are already 6h values; align_htf_to_ltf with df_1d is used to properly shift/index
    # but we actually want to align the 6h Ichimoku to 6h prices - so we should use identity alignment
    # Fix: since Ichimoku is calculated on 6h, we don't need HTF alignment for the values themselves
    # We only need to ensure we don't use future data - rolling with min_periods ensures this
    # So we use the raw calculated arrays directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup: max(52, 26+26 for Senkou shift, 50 for weekly EMA, 50 for ATR MA)
    start_idx = max(52, 52, 50, 50)  # 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_lag[i]) or np.isnan(senkou_b_lag[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_lag[i], senkou_b_lag[i])
        cloud_bottom = min(senkou_a_lag[i], senkou_b_lag[i])
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_bearish = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Weekly trend: uptrend if price > weekly EMA50, downtrend if price < weekly EMA50
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volatility filter: expanding volatility (ATR ratio > 1.2)
        vol_expanding = atr_ratio_aligned[i] > 1.2
        
        # Long logic: bullish TK cross + price above cloud + weekly uptrend + expanding volatility
        if tk_bullish and price_above_cloud and weekly_uptrend and vol_expanding:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish TK cross + price below cloud + weekly downtrend + expanding volatility
        elif tk_bearish and price_below_cloud and weekly_downtrend and vol_expanding:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses Tenkan-Kijun midpoint in opposite direction
        elif position == 1 and close[i] < (tenkan[i] + kijun[i]) / 2.0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (tenkan[i] + kijun[i]) / 2.0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_WeeklyTrend_1dVolatility"
timeframe = "6h"
leverage = 1.0