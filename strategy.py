#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku TK Cross with 1d Kumo filter and volume confirmation.
# Long when TK cross bullish (Tenkan > Kijun) AND price above 1d Kumo (Senkou Span A & B) AND 6h volume > 1.3x 20-period average.
# Short when TK cross bearish (Tenkan < Kijun) AND price below 1d Kumo AND 6h volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Ichimoku provides trend, momentum, and support/resistance in one system.
# 1d Kumo filter ensures alignment with higher timeframe trend, volume spike confirms participation.
# Designed to work in both bull (buy strength) and bear (sell weakness) markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

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
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): not used in signals (plots close 26 periods behind)
    
    # TK Cross: Tenkan - Kijun
    tk_cross = tenkan - kijun
    
    # === 6h Indicators: Volume Spike ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 1d data once before loop for Kumo filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need enough for Senkou Span B (52-period)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Indicators: Ichimoku Kumo (Cloud) ===
    # Tenkan-sen 1d
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen 1d
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A 1d
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B 1d
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Kumo top and bottom (Senkou Span A and B)
    kumO_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumO_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Kumo to 6h timeframe
    kumO_top_aligned = align_htf_to_ltf(prices, df_1d, kumO_top_1d)
    kumO_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumO_bottom_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 52 periods needed for Senkou B)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tk_cross[i]) or np.isnan(kumO_top_aligned[i]) or 
            np.isnan(kumO_bottom_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        tk = tk_cross[i]
        price = close[i]
        kumO_top = kumO_top_aligned[i]
        kumO_bottom = kumO_bottom_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if TK cross turns bearish or price falls below Kumo bottom or volume spike ends
            if tk <= 0 or price < kumO_bottom or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if TK cross turns bullish or price rises above Kumo top or volume spike ends
            if tk >= 0 or price > kumO_top or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: TK cross bullish (Tenkan > Kijun) AND price above Kumo AND volume spike
            if tk > 0 and price > kumO_top and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: TK cross bearish (Tenkan < Kijun) AND price below Kumo AND volume spike
            elif tk < 0 and price < kumO_bottom and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_IchimokuTKCross_1dKumoFilter_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0