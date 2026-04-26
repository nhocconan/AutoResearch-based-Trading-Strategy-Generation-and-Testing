#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_TK_Cross_v1
Hypothesis: On 6h timeframe, trade Ichimoku TK cross (Tenkan/Kijun) with 1d cloud filter (price above/below Kumo) and volume confirmation. Ichimoku provides trend, support/resistance, and momentum in one indicator. The 1d cloud acts as a strong trend filter to avoid counter-trend trades, while TK cross provides timely entries. Volume confirmation reduces false signals. Designed to work in both bull and bear markets by only taking trades aligned with the higher timeframe trend (cloud color and price vs cloud).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud (HTF trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_b = senkou_b.shift(26)
    
    # Current Kumo (cloud) boundaries: Senkou A and B
    # For trend filter: price above/below cloud
    # We need current cloud (not shifted) - so we use Senkou A/B that were calculated 26 periods ago
    # Actually, for current cloud, we use Senkou A/B without the forward shift
    # But standard Ichimoku plots Senkou A/B shifted 26 periods ahead
    # For cloud filter at current time, we use the Senkou A/B values that were plotted 26 periods ago
    # So we need to get the Senkou A/B values from 26 periods ago (i.e., unshifted)
    # Let's recalculate without shift for current cloud
    senkou_a_current = (tenkan_sen + kijun_sen) / 2
    senkou_b_current = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                        pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_current.values, senkou_b_current.values)
    cloud_bottom = np.minimum(senkou_a_current.values, senkou_b_current.values)
    
    # Price vs cloud: above cloud (bullish), below cloud (bearish), in cloud (neutral)
    price_above_cloud = close_1d > cloud_top
    price_below_cloud = close_1d < cloud_bottom
    
    # Get 6d data for TK cross (we'll calculate on 6h but need alignment)
    # Actually, we calculate TK cross on 6h directly for timely signals
    # But we need enough history
    if len(prices) < 26:
        return np.zeros(n)
    
    # Tenkan-sen (6h): (9-period high + 9-period low)/2
    tenkan_sen_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (6h): (26-period high + 26-period low)/2
    kijun_sen_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_sen_6h > kijun_sen_6h) & (tenkan_sen_6h.shift(1) <= kijun_sen_6h.shift(1))
    tk_cross_down = (tenkan_sen_6h < kijun_sen_6h) & (tenkan_sen_6h.shift(1) >= kijun_sen_6h.shift(1))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align HTF indicators (1d cloud) to 6h timeframe
    price_above_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_above_cloud.values)
    price_below_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_below_cloud.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of TK components (26), volume MA (20), Ichimoku (52)
    start_idx = max(26, 20, 52) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(price_above_cloud_aligned[i]) or 
            np.isnan(price_below_cloud_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or
            np.isnan(kijun_sen_6h[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        price_above = price_above_cloud_aligned[i]
        price_below = price_below_cloud_aligned[i]
        tk_up = tk_cross_up.iloc[i] if hasattr(tk_cross_up, 'iloc') else tk_cross_up[i]
        tk_down = tk_cross_down.iloc[i] if hasattr(tk_cross_down, 'iloc') else tk_cross_down[i]
        vol_spike = volume_spike.iloc[i] if hasattr(volume_spike, 'iloc') else volume_spike[i]
        
        if position == 0:
            # Long: TK cross up + price above 1d cloud + volume spike
            long_signal = tk_up and price_above and vol_spike
            
            # Short: TK cross down + price below 1d cloud + volume spike
            short_signal = tk_down and price_below and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down OR price falls below 1d cloud
            if tk_down or not price_above:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above 1d cloud
            if tk_up or not price_below:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_TK_Cross_v1"
timeframe = "6h"
leverage = 1.0