#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross with 1d ADX Trend Filter
# Long when: Tenkan-sen > Kijun-sen (TK Cross bullish) AND price > Kumo cloud (Senou Span A/B) AND 1d ADX > 25 (strong trend)
# Short when: Tenkan-sen < Kijun-sen (TK Cross bearish) AND price < Kumo cloud AND 1d ADX > 25
# Exit when TK Cross reverses OR price re-enters cloud
# Uses discrete sizing (0.25) to limit fee drag. Ichimoku provides dynamic support/resistance and trend identification.
# ADX filter ensures we only trade in trending markets, reducing whipsaws in ranging conditions.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_Ichimoku_TK_Cross_1dADX25_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    adx_1d = adx  # already aligned to 1d index
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    if len(high) >= period_tenkan:
        tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values +
                      pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values) / 2
    else:
        tenkan_sen = np.full(n, np.nan)
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    if len(high) >= period_kijun:
        kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values +
                     pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values) / 2
    else:
        kijun_sen = np.full(n, np.nan)
    
    # Senou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    # But for alignment, we calculate current value and shift later if needed
    senou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senou_b = 52
    if len(high) >= period_senou_b:
        senou_span_b = (pd.Series(high).rolling(window=period_senou_b, min_periods=period_senou_b).max().values +
                        pd.Series(low).rolling(window=period_senou_b, min_periods=period_senou_b).min().values) / 2
    else:
        senou_span_b = np.full(n, np.nan)
    
    # Kumo cloud boundaries (we use current cloud for price comparison)
    # Note: In real Ichimoku, Senou spans are plotted 26 periods ahead,
    # but for trend filtering we compare price to current cloud calculation
    kumo_top = np.maximum(senou_span_a, senou_span_b)
    kumo_bottom = np.minimum(senou_span_a, senou_span_b)
    
    # TK Cross signals
    tk_cross_bullish = tenkan_sen > kijun_sen
    tk_cross_bearish = tenkan_sen < kijun_sen
    
    # Price relative to cloud
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    # Trend filter: 1d ADX > 25
    strong_trend = adx_1d_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start loop after sufficient warmup for all indicators
    start_idx = max(52, 26, period_senou_b) + 10  # extra buffer for ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK Cross bullish AND price above cloud AND strong 1d trend
            if (tk_cross_bullish[i] and price_above_kumo[i] and strong_trend[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK Cross bearish AND price below cloud AND strong 1d trend
            elif (tk_cross_bearish[i] and price_below_kumo[i] and strong_trend[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK Cross bearish OR price re-enters cloud
            if (tk_cross_bearish[i] or not price_above_kumo[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK Cross bullish OR price re-enters cloud
            if (tk_cross_bullish[i] or not price_below_kumo[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals