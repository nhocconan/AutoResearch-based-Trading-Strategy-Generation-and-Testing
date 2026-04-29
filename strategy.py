#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross with 1d EMA200 trend filter and volume spike
# Long when: Tenkan-sen > Kijun-sen (TK Cross bullish) AND price > Cloud (bullish) AND price > 1d EMA200 AND volume > 2.0x 20-bar avg
# Short when: Tenkan-sen < Kijun-sen (TK Cross bearish) AND price < Cloud (bearish) AND price < 1d EMA200 AND volume > 2.0x 20-bar avg
# Exit when: TK Cross reverses OR price crosses Cloud middle (Senkou Span B)
# Uses discrete position sizing (0.25) to reduce fee drag.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid overtrading.
# Ichimoku provides strong trend confirmation; Cloud acts as dynamic support/resistance.
# TK Cross gives timely entries while 1d EMA200 ensures alignment with higher timeframe trend.
# Volume spike filters out low-momentum breakouts. Works in bull markets by capturing uptrends
# and in bear markets by shorting downtrends with trend alignment preventing counter-trend trades.

name = "6h_Ichimoku_Cloud_TK_Cross_1dEMA200_Volume_v1"
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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Ichimoku components on 6h data
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2.0
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2.0
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Cloud (Kumo): between Senkou Span A and B
    # Bullish cloud: Senkou Span A > Senkou Span B
    # Bearish cloud: Senkou Span A < Senkou Span B
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_tenkan, period_kijun, period_senkou_b, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema200_1d = ema_200_1d_aligned[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_span_a = senkou_span_a[i]
        curr_span_b = senkou_span_b[i]
        curr_close = close[i]
        
        # Determine cloud status
        bullish_cloud = curr_span_a > curr_span_b
        bearish_cloud = curr_span_a < curr_span_b
        cloud_middle = (curr_span_a + curr_span_b) / 2.0
        
        # TK Cross signals
        tk_bullish = curr_tenkan > curr_kijun
        tk_bearish = curr_tenkan < curr_kijun
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TK Cross turns bearish OR price crosses below cloud middle
            if tk_bearish or curr_close < cloud_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK Cross turns bullish OR price crosses above cloud middle
            if tk_bullish or curr_close > cloud_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when: TK Cross bullish AND price > Cloud (bullish) AND price > 1d EMA200 AND volume confirmation
            if tk_bullish and curr_close > max(curr_span_a, curr_span_b) and curr_close > curr_ema200_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when: TK Cross bearish AND price < Cloud (bearish) AND price < 1d EMA200 AND volume confirmation
            elif tk_bearish and curr_close < min(curr_span_a, curr_span_b) and curr_close < curr_ema200_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals