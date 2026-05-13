#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Enters long when price breaks above Kumo (cloud) with bullish TK cross, 1d bullish trend (close > EMA50), and volume > 1.8x MA20.
# Enters short when price breaks below Kumo with bearish TK cross, 1d bearish trend (close < EMA50), and volume > 1.8x MA20.
# Exits when TK cross reverses or price re-enters the cloud.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~12-37/year) to work in both bull and bear markets by requiring strong trend alignment and volume confirmation.
# Ichimoku is effective in trending markets (bull/bear) and avoids ranging periods via cloud filter.

name = "6h_Ichimoku_Kumo_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 6h data for Ichimoku components
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku parameters: Tenkan-sen (9), Kijun-sen (26), Senkou Span B (52)
    period_tenkan = 9
    period_kijun = 26
    period_senkou_b = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over 9 periods
    highest_9 = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_9 = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_9 + lowest_9) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over 26 periods
    highest_26 = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_26 = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_26 + lowest_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over 52 periods shifted 26 periods ahead
    highest_52 = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_52 = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_52 + lowest_52) / 2
    
    # Align Ichimoku components to 6h timeframe (already on 6h, but align for consistency)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Upper cloud = max(Senkou Span A, Senkou Span B)
    # Lower cloud = min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    lower_cloud = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # TK Cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross = tenkan_sen_aligned - kijun_sen_aligned  # >0: bullish, <0: bearish
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for all indicators (max lookback: 52 + 26 = 78)
    start_idx = max(period_senkou_b + period_kijun, 100)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper cloud with bullish TK cross, 1d bullish trend, and volume spike
            if (close[i] > upper_cloud[i] and 
                tk_cross[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower cloud with bearish TK cross, 1d bearish trend, and volume spike
            elif (close[i] < lower_cloud[i] and 
                  tk_cross[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross turns bearish OR price re-enters cloud (above lower cloud)
            if tk_cross[i] < 0 or close[i] < upper_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross turns bullish OR price re-enters cloud (below upper cloud)
            if tk_cross[i] > 0 or close[i] > lower_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals