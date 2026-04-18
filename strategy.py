# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h trend following with 1d Ichimoku cloud filter and volume confirmation.
# Long when price breaks above 6h Donchian(20) high with price > 1d Kumo (cloud) top and volume > 1.5x 24-period average.
# Short when price breaks below 6h Donchian(20) low with price < 1d Kumo bottom and same volume filter.
# Exit when price crosses back below/above 6h EMA(34) or Kumo flips.
# Uses Kumo as trend filter to avoid whipsaws in ranging markets, Donchian for breakouts, volume for conviction.
# Designed for ~15-30 trades/year per symbol (~60-120 total over 4 years).
name = "6h_IchimokuKumo_DonchianBreakout_VolumeFilter"
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
    
    # 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Kumo (cloud) top and bottom (current values, not shifted)
    # For cloud filtering, we use the current Senkou Span A and B values
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align Ichimoku cloud to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    
    # 6h Donchian channel (20 periods)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h EMA(34) for exit signal
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 6h = 6 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34[i]) or np.isnan(kumo_top_aligned[i]) or
            np.isnan(kumo_bottom_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_val = ema_34[i]
        kumo_top_val = kumo_top_aligned[i]
        kumo_bottom_val = kumo_bottom_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, above Kumo top, with volume
            if close_val > donch_high and close_val > kumo_top_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below Kumo bottom, with volume
            elif close_val < donch_low and close_val < kumo_bottom_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA(34) OR below Kumo bottom
            if close_val < ema_val or close_val < kumo_bottom_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA(34) OR above Kumo top
            if close_val > ema_val or close_val > kumo_top_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals