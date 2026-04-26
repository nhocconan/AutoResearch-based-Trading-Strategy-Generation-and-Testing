#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter and volume confirmation works in both bull and bear markets.
- Long when price breaks above Ichimoku cloud (Senkou Span A) AND 1d EMA50 uptrend AND volume > 2.0 * volume_ma(20)
- Short when price breaks below Ichimoku cloud (Senkou Span B) AND 1d EMA50 downtrend AND volume > 2.0 * volume_ma(20)
- Uses Ichimoku cloud (Senkou Span A/B) from 6h chart for dynamic support/resistance
- 1d EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike (2.0x) confirms institutional participation and reduces false breakouts
- Designed for low frequency (target 12-30 trades/year on 6h) to minimize fee drag
- Novelty: Ichimoku cloud provides adaptive support/resistance that adjusts to volatility, unlike fixed pivot levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter (needs completed 1d candle)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Ichimoku components on 6h chart (primary timeframe)
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Lagging Span (Chikou Span): Close plotted 26 periods behind (not used for breakout)
    # Cloud top/bottom: Senkou Span A/B form the cloud
    # For breakout: price above Senkou Span A (bullish) or below Senkou Span B (bearish)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou Span B, 50 for 1d EMA, 20 for volume MA)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku cloud breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Senkou Span A (cloud top) AND 1d uptrend AND volume spike
            if close[i] > senkou_span_a[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Senkou Span B (cloud bottom) AND 1d downtrend AND volume spike
            elif close[i] < senkou_span_b[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Senkou Span B (cloud bottom) OR 1d trend turns down
            if close[i] < senkou_span_b[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Senkou Span A (cloud top) OR 1d trend turns up
            if close[i] > senkou_span_a[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0