#!/usr/bin/env python3
"""
6h_IchimokuCloud_1dTrend_Volume
Hypothesis: Ichimoku cloud provides dynamic support/resistance with forward-looking Kumo. 
Buy when price breaks above Kumo (Tenkan > Kijun > Senkou Span A/B) in uptrend with volume.
Sell when price breaks below Kumo in downtrend with volume.
Uses 1d Ichimoku for trend filter (more stable) and 6h for entry timing.
Designed for 15-35 trades/year per symbol with clear trend-following logic.
Works in bull markets via breakouts and in bear via trend-following shorts.
"""

name = "6h_IchimokuCloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Ichimoku Cloud Calculation ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Current close shifted 26 periods back
    # Not used for signals but calculated for completeness
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # --- 6h Trend Indicators for Entry Timing ---
    # EMA20 for dynamic support/resistance
    ema20_6h = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- Volume Filter: spike above 1.3x median of last 30 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=30, min_periods=15).median().values
    vol_threshold = vol_median * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period (need 52 periods for Senkou Span B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema20_6h[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_6h[i] <= entry_price - 2.5 * (high_6h[i] - low_6h[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.5 * (high_6h[i] - low_6h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine Kumo (cloud) boundaries and direction
        # Kumo top is the higher of Senkou Span A and B
        # Kumo bottom is the lower of Senkou Span A and B
        kumo_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        kumo_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Kumo is bullish when Senkou A > Senkou B, bearish when Senkou A < Senkou B
        kumo_bullish = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        kumo_bearish = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # Price above/below Kumo
        price_above_kumo = close_6h[i] > kumo_top
        price_below_kumo = close_6h[i] < kumo_bottom
        
        # Tenkan/Kijun cross for momentum confirmation
        tenkan_above_kijun = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tenkan_below_kijun = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Volume filter
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Long entry: Price breaks above Kumo in bullish cloud with Tenkan/Kijun cross up and volume
            if price_above_kumo and kumo_bullish and tenkan_above_kijun and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            # Short entry: Price breaks below Kumo in bearish cloud with Tenkan/Kijun cross down and volume
            elif price_below_kumo and kumo_bearish and tenkan_below_kijun and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Manage existing position
            if position == 1:
                # Stoploss: 2.5x ATR (using high-low as proxy)
                if close_6h[i] <= entry_price - 2.5 * (high_6h[i] - low_6h[i]):
                    signals[i] = 0.0
                    position = 0
                # Exit: Price re-enters Kumo or Tenkan crosses below Kijun
                elif not price_above_kumo or tenkan_below_kijun:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss: 2.5x ATR
                if close_6h[i] >= entry_price + 2.5 * (high_6h[i] - low_6h[i]):
                    signals[i] = 0.0
                    position = 0
                # Exit: Price re-enters Kumo or Tenkan crosses above Kijun
                elif not price_below_kumo or tenkan_above_kijun:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals