#!/usr/bin/env python3
"""
4h_Ichimoku_Cloud_Trend_Momentum
Hypothesis: Ichimoku cloud provides robust trend direction and support/resistance zones.
Combined with momentum (RSI) and volume confirmation to filter false breakouts.
Designed to capture trending moves in both bull and bear markets while minimizing false signals
through multi-factor confirmation. Targets 20-40 trades/year to avoid fee drag.
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
    
    # Get daily data for Ichimoku calculation (more stable)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Ichimoku components on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    chikou_span = close_1d  # Will be shifted when aligning
    
    # Align Ichimoku components to 4h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span, additional_delay_bars=26)
    
    # Get 4h data for RSI and volume filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # RSI(14) on 4h for momentum confirmation
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio: current vs 20-period average
    volume_4h = df_4h['volume'].values
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_4h / np.where(vol_ma == 0, 1e-10, vol_ma)
    
    # Align 4h indicators to main timeframe (should be 1:1 but using align for safety)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_4h, volume_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure Ichimoku and RSI are stable
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(volume_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku trend: price above/below cloud
        above_cloud = (close[i] > senkou_span_a_aligned[i]) and (close[i] > senkou_span_b_aligned[i])
        below_cloud = (close[i] < senkou_span_a_aligned[i]) and (close[i] < senkou_span_b_aligned[i])
        
        # Price vs Kijun-sen for momentum
        price_above_kijun = close[i] > kijun_sen_aligned[i]
        price_below_kijun = close[i] < kijun_sen_aligned[i]
        
        # Tenkan/Kijun cross for momentum
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Chikou confirmation: price vs price 26 periods ago
        chikou_confirm_long = close[i] > chikou_span_aligned[i]
        chikou_confirm_short = close[i] < chikou_span_aligned[i]
        
        # RSI momentum filter: avoid overbought/oversold extremes
        rsi_momentum_long = (rsi_aligned[i] > 50) and (rsi_aligned[i] < 70)
        rsi_momentum_short = (rsi_aligned[i] < 50) and (rsi_aligned[i] > 30)
        
        # Volume confirmation: above average volume
        volume_confirm = volume_ratio_aligned[i] > 1.2
        
        # Entry conditions: Strong alignment of trend, momentum, and volume
        long_entry = (above_cloud and 
                     price_above_kijun and 
                     tk_cross_bullish and 
                     chikou_confirm_long and 
                     rsi_momentum_long and 
                     volume_confirm)
                     
        short_entry = (below_cloud and 
                      price_below_kijun and 
                      tk_cross_bearish and 
                      chikou_confirm_short and 
                      rsi_momentum_short and 
                      volume_confirm)
        
        # Exit conditions: Cloud reversal or momentum loss
        long_exit = (not above_cloud) or (not price_above_kijun) or (not tk_cross_bullish)
        short_exit = (not below_cloud) or (not price_below_kijun) or (not tk_cross_bearish)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Ichimoku_Cloud_Trend_Momentum"
timeframe = "4h"
leverage = 1.0