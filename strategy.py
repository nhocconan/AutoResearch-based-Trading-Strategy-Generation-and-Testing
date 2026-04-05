#!/usr/bin/env python3
"""
Experiment #7611: 6h Ichimoku Cloud with 1-day trend filter.
Hypothesis: In bull markets (price > 1d Tenkan-Sen), go long when TK crosses above and price above cloud.
In bear markets (price < 1d Tenkan-Sen), go short when TK crosses below and price below cloud.
This combines trend-following with momentum and avoids whipsaws in sideways markets.
Targets 50-150 total trades over 4 years (12-37/year) with strict Ichimoku conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7611_6h_ichimoku_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_SEN = 9
KIJUN_SEN = 26
SENKOU_SPAN_B = 52
DISPLACEMENT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < SENKOU_SPAN_B + DISPLACEMENT:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=TENKAN_SEN, min_periods=TENKAN_SEN).max() + 
                     pd.Series(low_1d).rolling(window=TENKAN_SEN, min_periods=TENKAN_SEN).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=KIJUN_SEN, min_periods=KIJUN_SEN).max() + 
                    pd.Series(low_1d).rolling(window=KIJUN_SEN, min_periods=KIJUN_SEN).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a_1d = (tenkan_sen_1d + kijun_sen_1d) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=SENKOU_SPAN_B, min_periods=SENKOU_SPAN_B).max() + 
                        pd.Series(low_1d).rolling(window=SENKOU_SPAN_B, min_periods=SENKOU_SPAN_B).min()) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    # Calculate LTF Ichimoku components for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line)
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_SEN, min_periods=TENKAN_SEN).max() + 
                  pd.Series(low).rolling(window=TENKAN_SEN, min_periods=TENKAN_SEN).min()) / 2
    # Kijun-sen (Base Line)
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_SEN, min_periods=KIJUN_SEN).max() + 
                 pd.Series(low).rolling(window=KIJUN_SEN, min_periods=KIJUN_SEN).min()) / 2
    # Senkou Span A (Leading Span A)
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B)
    senkou_span_b = (pd.Series(high).rolling(window=SENKOU_SPAN_B, min_periods=SENKOU_SPAN_B).max() + 
                     pd.Series(low).rolling(window=SENKOU_SPAN_B, min_periods=SENKOU_SPAN_B).min()) / 2
    
    # Calculate ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_SPAN_B + DISPLACEMENT, KIJUN_SEN) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine 1d trend regime using Tenkan-sen
        bull_regime = close[i] > tenkan_sen_1d_aligned[i]   # price above 1d Tenkan-sen
        bear_regime = close[i] < tenkan_sen_1d_aligned[i]   # price below 1d Tenkan-sen
        
        # Ichimoku signals on 6h
        tk_cross_above = (tenkan_sen[i-1] <= kijun_sen[i-1]) and (tenkan_sen[i] > kijun_sen[i]) and (i-1 >= 0)
        tk_cross_below = (tenkan_sen[i-1] >= kijun_sen[i-1]) and (tenkan_sen[i] < kijun_sen[i]) and (i-1 >= 0)
        
        # Cloud boundaries (using displaced Senkou Span)
        # Note: Senkou Span is plotted 26 periods ahead, so we use current values for cloud
        upper_cloud = np.maximum(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        # Price above/below cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Entry conditions
        long_entry = bull_regime and tk_cross_above and price_above_cloud
        short_entry = bear_regime and tk_cross_below and price_below_cloud
        
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