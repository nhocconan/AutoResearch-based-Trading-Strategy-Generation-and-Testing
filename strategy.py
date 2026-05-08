#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND 1d close > 1d EMA50 (uptrend) AND 6h volume > 1.5x 20-period average.
# Short when price breaks below Kumo AND Tenkan < Kijun AND 1d close < 1d EMA50 AND 6h volume > 1.5x 20-period average.
# Exit when price re-enters the Kumo.
# Uses Ichimoku for trend/momentum, 1d EMA50 for higher timeframe trend filter, and volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "6h_Ichimoku_Kumo_TK_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA50 trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values + 
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values + 
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values + 
                      pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used in signals)
    
    # Align Ichimoku components to current time (account for Senkou Span shift)
    # Senkou Span A and B are plotted 26 periods ahead, so to get current cloud values,
    # we need to look at values that were calculated 26 periods ago
    senkou_span_a_current = np.roll(senkou_span_a, 26)
    senkou_span_b_current = np.roll(senkou_span_b, 26)
    # First 26 values will be invalid due to roll, handled by NaN check
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    kumo_top = np.maximum(senkou_span_a_current, senkou_span_b_current)
    kumo_bottom = np.minimum(senkou_span_a_current, senkou_span_b_current)
    
    # TK Cross: Tenkan > Kijun for bullish, Tenkan < Kijun for bearish
    tk_cross_bullish = tenkan_sen > kijun_sen
    tk_cross_bearish = tenkan_sen < kijun_sen
    
    # Daily EMA50 for trend filter
    close_d = df_d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    daily_uptrend = close_d > ema50_d  # Daily close above EMA50
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_d, daily_uptrend)
    daily_downtrend = close_d < ema50_d  # Daily close below EMA50
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_d, daily_downtrend)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # Sufficient warmup for Senkou Span B (52-period)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(tk_cross_bullish[i]) or np.isnan(tk_cross_bearish[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above Kumo, bullish TK cross, daily uptrend, volume confirmation
            long_cond = (close[i] > kumo_top[i]) and tk_cross_bullish[i] and daily_uptrend_aligned[i] and volume_filter[i]
            # Short conditions: price below Kumo, bearish TK cross, daily downtrend, volume confirmation
            short_cond = (close[i] < kumo_bottom[i]) and tk_cross_bearish[i] and daily_downtrend_aligned[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters Kumo (closes below Kumo top)
            if close[i] < kumo_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Kumo (closes above Kumo bottom)
            if close[i] > kumo_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals