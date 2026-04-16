#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 1d Kumo (cloud) filter and volume confirmation.
# Long when Tenkan crosses above Kijun AND price > 1d Kumo top AND 6h volume > 1.5x 20-period average.
# Short when Tenkan crosses below Kijun AND price < 1d Kumo bottom AND 6h volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Ichimoku provides trend/momentum signals, 1d Kumo acts as dynamic support/resistance filter,
# volume spike confirms institutional participation. Designed to capture trends while avoiding false breakouts in ranging markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Ichimoku Components ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # === 6h Indicators: Volume Spike ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for Kumo (cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Senkou Span B (52-period)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Indicators: Ichimoku Kumo (Cloud) ===
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    # But for cloud filter, we use current values: (1d Tenkan + 1d Kijun)/2
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_span_a = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high_1d + period52_low_1d) / 2
    
    # Kumo top/bottom (current cloud boundaries)
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align 1d Kumo to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 52 periods needed for Senkou B)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        price = close[i]
        kumo_top_val = kumo_top_aligned[i]
        kumo_bottom_val = kumo_bottom_aligned[i]
        vol_spike = volume_spike[i]
        
        # Ichimoku cross detection (using previous bar to avoid look-ahead)
        tenkan_prev = tenkan[i-1]
        kijun_prev = kijun[i-1]
        
        bullish_cross = (tenkan_prev <= kijun_prev) and (tenkan_val > kijun_val)
        bearish_cross = (tenkan_prev >= kijun_prev) and (tenkan_val < kijun_val)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if bearish cross OR price falls below Kumo bottom OR volume spike ends
            if bearish_cross or price < kumo_bottom_val or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if bullish cross OR price rises above Kumo top OR volume spike ends
            if bullish_cross or price > kumo_top_val or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bullish TK cross AND price > Kumo top AND volume spike
            if bullish_cross and price > kumo_top_val and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bearish TK cross AND price < Kumo bottom AND volume spike
            elif bearish_cross and price < kumo_bottom_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_IchimokuTKCross_1dKumoFilter_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0