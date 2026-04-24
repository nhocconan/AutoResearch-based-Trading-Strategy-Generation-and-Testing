#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud (Tenkan/Kijun/Senkou) with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for trend filter (price above/below Kumo cloud) and Senkou span alignment.
- Entry: Long when Tenkan > Kijun (bullish TK cross) AND price > Kumo (cloud) AND 1d Senkou A > Senkou B (bullish cloud) AND volume > 1.5x 20-period MA.
         Short when Tenkan < Kijun (bearish TK cross) AND price < Kumo (cloud) AND 1d Senkou A < Senkou B (bearish cloud) AND volume > 1.5x 20-period MA.
- Exit: Opposite TK cross OR price crosses Kumo in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Ichimoku identifies trend, momentum, and support/resistance via multiple lines.
- Volume confirmation ensures breakouts have conviction.
- Works in bull markets (buy when bullish aligned) and bear markets (sell when bearish aligned).
- Estimated trades: ~100 total over 4 years (~25/year) based on TK cross frequency with trend/volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HTF data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for Senkou span calculation
        return np.zeros(n)
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe (with proper delay for completed bars)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # Senkou is already leading, align properly
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Kumo (Cloud) boundaries: Senkou Span A and Senkou Span B
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # 6h Ichimoku for entry signals (Tenkan/Kijun cross)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h_entry = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h_entry = (period26_high_6h + period26_low_6h) / 2
    
    # TK Cross signals
    tk_bullish = tenkan_6h_entry > kijun_6h_entry  # Tenkan above Kijun = bullish
    tk_bearish = tenkan_6h_entry < kijun_6h_entry  # Tenkan below Kijun = bearish
    
    # Price vs Cloud (6h)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # 1d Cloud trend (bullish/bearish cloud)
    cloud_bullish = senkou_a_6h > senkou_b_6h  # Senkou A above Senkou B = bullish cloud
    cloud_bearish = senkou_a_6h < senkou_b_6h  # Senkou A below Senkou B = bearish cloud
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(tenkan_6h_entry[i]) or np.isnan(kijun_6h_entry[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite TK cross OR price crosses Kumo in opposite direction
        if position != 0:
            # Exit long: bearish TK cross OR price falls below cloud
            if position == 1:
                if tk_bearish[i] or curr_close < cloud_bottom[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish TK cross OR price rises above cloud
            elif position == -1:
                if tk_bullish[i] or curr_close > cloud_top[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction with volume confirmation
        if position == 0:
            # Long: bullish TK cross AND price above cloud AND bullish cloud AND volume confirmation
            if tk_bullish[i] and price_above_cloud[i] and cloud_bullish[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross AND price below cloud AND bearish cloud AND volume confirmation
            elif tk_bearish[i] and price_below_cloud[i] and cloud_bearish[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0