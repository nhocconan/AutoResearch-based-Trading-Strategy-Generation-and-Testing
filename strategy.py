#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Regime
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 6h with 1d trend filter (price > EMA50) and low volatility regime (ATR ratio < 0.8) captures trend acceleration after consolidation. Works in bull/bear via 1d EMA filter and volatility regime avoidance. Targets 12-25 trades/year by requiring cloud twist + trend + low vol confluence.
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
    
    # Get 1d data for HTF trend and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) on 1d for volatility regime
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Ichimoku components on 6h (conversion line, base line, leading spans)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align Ichimoku components (no additional delay needed as they are based on completed periods)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan_sen)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b)
    
    # Cloud twist detection: Senkou Span A crosses above/below Senkou Span B
    # Bullish twist: Senkou A crosses above Senkou B (previous A <= previous B and current A > current B)
    # Bearish twist: Senkou A crosses below Senkou B (previous A >= previous B and current A < current B)
    senkou_a_prev = np.concatenate([[np.nan], senkou_a_aligned[:-1]])
    senkou_b_prev = np.concatenate([[np.nan], senkou_b_aligned[:-1]])
    
    bullish_twist = (senkou_a_prev <= senkou_b_prev) & (senkou_a_aligned > senkou_b_aligned)
    bearish_twist = (senkou_a_prev >= senkou_b_prev) & (senkou_a_aligned < senkou_b_aligned)
    
    # Volatility regime: ATR ratio (current ATR / 20-period average ATR) < 0.8 = low volatility
    atr_ma_20 = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d_aligned / atr_ma_20
    low_volatility = atr_ratio < 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, 26, 52, 20)  # EMA50, Kijun, Senkou B, ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        close_val = close[i]
        bullish_twist_val = bullish_twist[i]
        bearish_twist_val = bearish_twist[i]
        low_vol_val = low_volatility[i]
        
        if position == 0:
            # Look for entry signals: Ichimoku cloud twist with trend and low volatility
            # Long: bullish twist + price above 1d EMA50 (uptrend) + low volatility
            long_signal = bullish_twist_val and (close_val > ema_val) and low_vol_val
            # Short: bearish twist + price below 1d EMA50 (downtrend) + low volatility
            short_signal = bearish_twist_val and (close_val < ema_val) and low_vol_val
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif bearish_twist_val and (close_val < ema_val) and low_vol_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price crosses below Kijun-sen (base line) - trend weakening
            if close_val < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            # 2. Bearish cloud twist (exit long)
            elif bearish_twist_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price crosses above Kijun-sen (base line) - trend weakening
            if close_val > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            # 2. Bullish cloud twist (exit short)
            elif bullish_twist_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Regime"
timeframe = "6h"
leverage = 1.0