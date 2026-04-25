#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter and volume confirmation.
Ichimoku provides objective trend/momentum signals: TK cross above/below cloud indicates strong momentum.
Using 1d cloud as trend filter ensures alignment with higher timeframe direction.
Volume confirmation reduces false signals. Targets 12-37 trades/year by requiring confluence of:
1) TK cross, 2) price outside cloud (confirming trend strength), 3) 1d cloud color (trend filter),
4) volume > 1.5x 20-period average. Works in bull/bear markets as Ichimoku adapts to volatility.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Ichimoku cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displaced)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_period = 9
    high_tenkan = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_period = 26
    high_kijun = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    low_kijun = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 displaced 26 periods ahead
    senkou_period = 52
    high_senkou = pd.Series(high_1d).rolling(window=senkou_period, min_periods=senkou_period).max().values
    low_senkou = pd.Series(low_1d).rolling(window=senkou_period, min_periods=senkou_period).min().values
    senkou_b = ((high_senkou + low_senkou) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Cloud color: green if Senkou A > Senkou B (bullish), red otherwise
        cloud_bullish = senkou_a_aligned[i] > senkou_b_aligned[i]
        
        # TK cross conditions
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        # Price position relative to cloud
        price_above_cloud = curr_close > upper_cloud
        price_below_cloud = curr_close < lower_cloud
        price_in_cloud = (curr_close >= lower_cloud) and (curr_close <= upper_cloud)
        
        if position == 0:
            # Look for entry signals
            # Long: TK cross up, price above cloud, cloud bullish, volume confirmation
            long_signal = tk_cross_up and price_above_cloud and cloud_bullish and volume_confirm[i]
            # Short: TK cross down, price below cloud, cloud bearish, volume confirmation
            short_signal = tk_cross_down and price_below_cloud and (not cloud_bullish) and volume_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when TK cross down OR price enters cloud
            if tk_cross_down or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TK cross up OR price enters cloud
            if tk_cross_up or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0