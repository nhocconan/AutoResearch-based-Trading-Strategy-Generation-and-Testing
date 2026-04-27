#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike_v1
Hypothesis: Use Ichimoku cloud (TK cross + price vs cloud) from 1d timeframe as trend filter on 6h chart.
Enter long when price breaks above 6h Donchian(20) high + bullish TK cross on 1d + price above 1d cloud + volume spike.
Enter short when price breaks below 6h Donchian(20) low + bearish TK cross on 1d + price below 1d cloud + volume spike.
Ichimoku cloud provides strong trend identification with built-in support/resistance, reducing false breakouts.
Designed for low trade frequency (target: 12-30/year) to minimize fee drag. Works in both bull and bear markets
by aligning with higher timeframe trend. Uses discrete position sizing (0.25) to reduce churn.
Adds ATR-based stoploss to manage risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h Donchian(20) for breakout levels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # Leading span needs extra delay
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # Leading span needs extra delay
    
    # TK cross: Tenkan crossing above/below Kijun
    tk_cross_above = tenkan_aligned > kijun_aligned
    tk_cross_below = tenkan_aligned < kijun_aligned
    
    # Price vs Cloud: price above Senkou Span A AND Senkou Span B = bullish cloud
    # price below Senkou Span A AND Senkou Span B = bearish cloud
    bullish_cloud = (close_1d > senkou_a) & (close_1d > senkou_b)
    bearish_cloud = (close_1d < senkou_a) & (close_1d < senkou_b)
    bullish_cloud_aligned = align_htf_to_ltf(prices, df_1d, bullish_cloud.astype(float), additional_delay_bars=26)
    bearish_cloud_aligned = align_htf_to_ltf(prices, df_1d, bearish_cloud.astype(float), additional_delay_bars=26)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for ATR, Donchian, Ichimoku and volume average
    start_idx = max(100, 20, 52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(bullish_cloud_aligned[i]) or np.isnan(bearish_cloud_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: Donchian breakout in direction of 1d Ichimoku trend with volume spike
            # Long: price breaks above 6h Donchian high AND bullish TK cross AND price above 1d cloud AND volume spike
            # Short: price breaks below 6h Donchian low AND bearish TK cross AND price below 1d cloud AND volume spike
            long_breakout = close_val > donch_high[i]
            short_breakout = close_val < donch_low[i]
            bullish_tk = tk_cross_above[i]
            bearish_tk = tk_cross_below[i]
            bullish_cloud_signal = bullish_cloud_aligned[i] > 0.5
            bearish_cloud_signal = bearish_cloud_aligned[i] > 0.5
            
            if long_breakout and bullish_tk and bullish_cloud_signal and volume_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and bearish_tk and bearish_cloud_signal and volume_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below 6h Donchian low (failed breakout) or ATR stoploss hit
            if close_val < donch_low[i] or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above 6h Donchian high (failed breakout) or ATR stoploss hit
            if close_val > donch_high[i] or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0