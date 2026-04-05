#!/usr/bin/env python3
"""
Experiment #11215: 6h Ichimoku Cloud with Weekly Trend Filter
Hypothesis: Ichimoku Cloud (Tenkan/Kijun cross + cloud filter) on 6h provides high-probability entries.
Weekly trend filter (price vs weekly cloud) ensures alignment with higher timeframe momentum.
Works in bull markets via cloud breakouts and in bear markets via rejection from cloud resistance.
Targets 50-150 trades over 4 years (12-37/year) with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11215_6h_ichimoku_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
WEEKLY_TENKAN = 9
WEEKLY_KIJUN = 26
WEEKLY_SENKOU_B = 52
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(KIJUN_PERIOD)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou = pd.Series(close).shift(-KIJUN_PERIOD)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku for trend filter
    wk_tenkan, wk_kijun, wk_senkou_a, wk_senkou_b, _ = calculate_ichimoku(
        df_weekly['high'].values, 
        df_weekly['low'].values, 
        df_weekly['close'].values
    )
    # Weekly trend: price above/both Senkou spans = uptrend, below both = downtrend
    wk_trend_up = (df_weekly['close'].values > wk_senkou_a) & (df_weekly['close'].values > wk_senkou_b)
    wk_trend_down = (df_weekly['close'].values < wk_senkou_a) & (df_weekly['close'].values < wk_senkou_b)
    wk_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, wk_trend_up.astype(float))
    wk_trend_down_aligned = align_htf_to_ltf(prices, df_weekly, wk_trend_down.astype(float))
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (need enough data for Ichimoku calculations)
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD) + KIJUN_PERIOD
    
    for i in range(start, n):
        # Skip if weekly trend data not available
        if np.isnan(wk_trend_up_aligned[i]) or np.isnan(wk_trend_down_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Ichimoku signals
        # Tenkan/Kijun cross
        tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1]) if i > 0 else False
        tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1]) if i > 0 else False
        
        # Price relative to cloud
        price_above_cloud = (close[i] > senkou_a[i]) and (close[i] > senkou_b[i])
        price_below_cloud = (close[i] < senkou_a[i]) and (close[i] < senkou_b[i])
        
        # Chikou confirmation (price vs past close)
        chikou_confirm = False
        if i >= KIJUN_PERIOD and not np.isnan(chikou[i-KIJUN_PERIOD]):
            chikou_confirm = close[i-KIJUN_PERIOD] > close[i]  # Chikou above current price = bullish
        
        # Entry conditions
        long_entry = tk_cross_up and price_above_cloud and wk_trend_up_aligned[i] > 0.5
        short_entry = tk_cross_down and price_below_cloud and wk_trend_down_aligned[i] > 0.5
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals