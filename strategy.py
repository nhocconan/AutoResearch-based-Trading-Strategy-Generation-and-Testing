#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for trend (price > Kumo) and volume average.
- Ichimoku Components:
  * Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
  * Kijun-sen (Base Line): (26-period high + 26-period low)/2
  * Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
  * Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
  * Kumo (Cloud): between Senkou Span A and B
- Entry: Long when price breaks above Kumo AND Tenkan > Kijun (bullish TK cross) AND volume > 1.5 * 1d average volume.
         Short when price breaks below Kumo AND Tenkan < Kijun (bearish TK cross) AND volume > 1.5 * 1d average volume.
- Exit: Opposite Ichimoku signal (price breaks Kumo in opposite direction).
- Signal size: 0.25 discrete to minimize fee drag.
- Ichimoku Cloud acts as dynamic support/resistance; breakouts with TK cross confirm momentum.
- Volume confirmation filters weak breakouts.
- Works in bull markets (catching trends) and bear markets (shorting breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan = (high_series.rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
              low_series.rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun = (high_series.rolling(window=period_kijun, min_periods=period_kijun).max() + 
             low_series.rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    senkou_b = (high_series.rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                low_series.rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 80:  # Need sufficient data for Ichimoku calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Ichimoku from 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 52  # Need 52 for Senkou Span B calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
        # For current period i, we need Senkou values from i-26 (already plotted)
        idx_kumo = i - 26
        if idx_kumo < 0:
            # Not enough data to plot cloud yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        senkou_a_kumo = senkou_a[idx_kumo]
        senkou_b_kumo = senkou_b[idx_kumo]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_kumo, senkou_b_kumo)
        cloud_bottom = min(senkou_a_kumo, senkou_b_kumo)
        
        # TK Cross conditions
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        # Volume confirmation
        volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
        
        # Exit conditions: opposite Ichimoku signal (price breaks Kumo in opposite direction)
        if position != 0:
            # Exit long: price breaks below cloud bottom
            if position == 1:
                if curr_low < cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above cloud top
            elif position == -1:
                if curr_high > cloud_top:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku breakout with TK cross and volume confirmation
        if position == 0:
            # Bullish breakout: price breaks above cloud top + bullish TK cross
            bullish_breakout = curr_high > cloud_top and close[i-1] <= cloud_top and tk_bullish
            # Bearish breakout: price breaks below cloud bottom + bearish TK cross
            bearish_breakout = curr_low < cloud_bottom and close[i-1] >= cloud_bottom and tk_bearish
            
            if bullish_breakout and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudBreakout_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0