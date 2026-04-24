#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA trend filter (bull/bear regime) and 1w for Ichimoku baseline confirmation.
- Ichimoku Components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement).
- Entry: Long when price > Cloud AND Tenkan > Kijun AND 1d EMA50 > 1d EMA200 AND volume > 1.5 * 20-period average volume.
         Short when price < Cloud AND Tenkan < Kijun AND 1d EMA50 < 1d EMA200 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Ichimoku signal (price crosses into/through Cloud or TK cross reverses).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1d EMA trend (EMA50 > EMA200 = bull, < = bear) and only trading Ichimoku signals in that direction, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for Ichimoku calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA trend to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values +
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values +
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values +
                      pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for entry/exit)
    
    # The Ichimoku Cloud is between Senkou Span A and Senkou Span B
    # For simplicity, we'll use the unshifted Senkou Span A/B to represent current cloud
    # (In practice, the cloud is plotted 26 periods ahead, but for signal generation we use current values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period_kijun, period_senkou_b, 200)  # Need 26 for Kijun, 52 for Senkou B, 200 for EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 > EMA200 = bullish trend, < = bearish trend
        bullish_trend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        bearish_trend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Ichimoku signals
        # Price above Cloud: close > max(Senkou Span A, Senkou Span B)
        # Price below Cloud: close < min(Senkou Span A, Senkou Span B)
        # Tenkan-sen > Kijun-sen = bullish momentum
        # Tenkan-sen < Kijun-sen = bearish momentum
        
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        tenkan_above_kijun = tenkan_sen[i] > kijun_sen[i]
        tenkan_below_kijun = tenkan_sen[i] < kijun_sen[i]
        
        # Exit conditions: opposite Ichimoku signal
        if position != 0:
            # Exit long: price falls below Cloud OR Tenkan crosses below Kijun
            if position == 1:
                if (curr_close < cloud_top or tenkan_below_kijun):
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above Cloud OR Tenkan crosses above Kijun
            elif position == -1:
                if (curr_close > cloud_bottom or tenkan_above_kijun):
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku breakout with trend and volume filters
        if position == 0:
            # Long: price > Cloud AND Tenkan > Kijun AND bullish trend AND volume confirmation
            long_condition = (price_above_cloud and
                            tenkan_above_kijun and
                            bullish_trend and
                            volume_confirm)
            
            # Short: price < Cloud AND Tenkan < Kijun AND bearish trend AND volume confirmation
            short_condition = (price_below_cloud and
                             tenkan_below_kijun and
                             bearish_trend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMATrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0