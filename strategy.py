#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Kumo twist and volume confirmation.
# Long when: Tenkan-sen > Kijun-sen (TK cross bullish) AND price > Senkou Span A (above cloud) AND 1d Senkou Span B rising (bullish Kumo) AND volume > 1.3x 20-period average.
# Short when: Tenkan-sen < Kijun-sen (TK cross bearish) AND price < Senkou Span A (below cloud) AND 1d Senkou Span B falling (bearish Kumo) AND volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Ichimoku provides trend, momentum, and support/resistance in one system. 1d Kumo twist ensures alignment with higher timeframe trend. Volume confirms participation.
# Designed to work in both bull (buy TK cross above cloud) and bear (sell TK cross below cloud) markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Ichimoku Components ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # === 6h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 1d data once before loop for Kumo twist
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Senkou Span B calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Ichimoku Components for Kumo twist ===
    # Tenkan-sen (1d)
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen (1d)
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A (1d)
    senkou_span_a_1d = (tenkan_sen_1d + kijun_sen_1d) / 2
    
    # Senkou Span B (1d): (52-period high + 52-period low) / 2
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Kumo twist: Senkou Span B rising/falling (bullish/bearish Kumo)
    # Use 3-period change to determine direction
    senkou_span_b_1d_rising = senkou_span_b_1d > np.roll(senkou_span_b_1d, 3)
    senkou_span_b_1d_falling = senkou_span_b_1d < np.roll(senkou_span_b_1d, 3)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    senkou_span_b_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d_rising.astype(float))
    senkou_span_b_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d_falling.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 52 periods needed)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i]) or
            np.isnan(senkou_span_b_1d_rising_aligned[i]) or np.isnan(senkou_span_b_1d_falling_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        span_a = senkou_span_a[i]
        span_b = senkou_span_b[i]
        tenkan_1d = tenkan_sen_1d_aligned[i]
        kijun_1d = kijun_sen_1d_aligned[i]
        span_a_1d = senkou_span_a_1d_aligned[i]
        span_b_1d = senkou_span_b_1d_aligned[i]
        span_b_1d_rising = senkou_span_b_1d_rising_aligned[i] > 0.5
        span_b_1d_falling = senkou_span_b_1d_falling_aligned[i] > 0.5
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if TK cross turns bearish or price falls below cloud or volume spike ends
            if tenkan < kijun or price < span_a or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if TK cross turns bullish or price rises above cloud or volume spike ends
            if tenkan > kijun or price > span_a or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: TK cross bullish AND price above cloud AND 1d Senkou Span B rising (bullish Kumo) AND volume spike
            if tenkan > kijun and price > span_a and span_b_1d_rising and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: TK cross bearish AND price below cloud AND 1d Senkou Span B falling (bearish Kumo) AND volume spike
            elif tenkan < kijun and price < span_a and span_b_1d_falling and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_IchimokuTKCross_1dKumoTwist_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0