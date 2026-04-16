#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Long when price > 1d Ichimoku cloud (Senou Span A/B) AND Tenkan > Kijun (bullish TK cross) AND volume > 1.5x 20-period 6h average.
# Short when price < 1d Ichimoku cloud AND Tenkan < Kijun (bearish TK cross) AND volume > 1.5x 20-period 6h average.
# Exit when TK cross reverses or price re-enters the cloud.
# Uses discrete position size 0.25. Designed to capture medium-term trends with momentum confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# Works in bull markets via cloud breakouts and in bear markets via short signals below cloud.
# Ichimoku provides dynamic support/resistance; TK cross adds momentum filter; volume confirms conviction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Volume and Ichimoku Components (Tenkan, Kijun) ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1d Indicators: Ichimoku Cloud (Senou Span A, Senou Span B) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Senou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    # But for cloud calculation, we use current values aligned to present
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senkou_span_a = ((tenkan_1d + kijun_1d) / 2)  # Not shifted for alignment - align_htf_to_ltf handles timing
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud boundaries: upper cloud = max(Senou A, Senou B), lower cloud = min(Senou A, Senou B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 52 periods needed for Senou B)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(volume_spike[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        tk_bullish = tenkan[i] > kijun[i]   # Bullish TK cross
        tk_bearish = tenkan[i] < kijun[i]   # Bearish TK cross
        price_above_cloud = price > cloud_top[i]
        price_below_cloud = price < cloud_bottom[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if TK cross turns bearish OR price re-enters cloud
            if not tk_bullish or (price > cloud_bottom[i] and price < cloud_top[i]):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if TK cross turns bullish OR price re-enters cloud
            if not tk_bearish or (price > cloud_bottom[i] and price < cloud_top[i]):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above cloud AND bullish TK cross AND volume spike
            if price_above_cloud and tk_bullish and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price below cloud AND bearish TK cross AND volume spike
            elif price_below_cloud and tk_bearish and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_IchimokuCloud_TKCross_1dFilter_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0