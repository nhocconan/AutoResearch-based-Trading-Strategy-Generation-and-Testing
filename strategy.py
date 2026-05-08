#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku cloud breakout with 1d weekly trend filter and volume confirmation
# Long when price breaks above Kumo (cloud) and TK cross is bullish, 1d EMA50 rising, volume > 2x average
# Short when price breaks below Kumo (cloud) and TK cross is bearish, 1d EMA50 falling, volume > 2x average
# Uses Ichimoku cloud for dynamic support/resistance, TK cross for momentum, EMA50 for trend, volume for confirmation
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag and high win rate

name = "6h_IchimokuCloud_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need at least 52 periods for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        senkou_a = senkou_span_a_aligned[i]
        senkou_b = senkou_span_b_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a, senkou_b)
        lower_cloud = min(senkou_a, senkou_b)
        
        # TK Cross: Tenkan-sen > Kijun-sen = bullish, < = bearish
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        if position == 0:
            # Enter long: price breaks above cloud, TK bullish, 1d uptrend, volume confirmation
            if low_val > upper_cloud and tk_bullish and ema50_val > close_val and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below cloud, TK bearish, 1d downtrend, volume confirmation
            elif high_val < lower_cloud and tk_bearish and ema50_val < close_val and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below cloud or TK turns bearish or 1d trend down
            if high_val < lower_cloud or not tk_bullish or ema50_val < close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above cloud or TK turns bullish or 1d trend up
            if low_val > upper_cloud or not tk_bearish or ema50_val > close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals