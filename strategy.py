#!/usr/bin/env python3
"""
6h_Ichimoku_KumoTwist_1dTrend_WeeklyVolume_v1
Hypothesis: Ichimoku TK cross with Kumo twist (Senkou A/B cross) from 1d timeframe, filtered by 1w EMA50 trend and volume confirmation. Captures strong trend reversals in both bull/bear markets by combining multiple confluence factors. Targets 12-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
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
    
    # Align Ichimoku components to 6h timeframe (with proper shift for forward-looking spans)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo twist: Senkou A crosses above/below Senkou Bullish when Senkou A > Senkou B
    # We need to detect the cross - bullish when A crosses above B, bearish when A crosses below B
    # For entry, we use the current relationship plus cross confirmation
    kumo_bullish = senkou_a_aligned > senkou_b_aligned
    kumo_bearish = senkou_a_aligned < senkou_b_aligned
    
    # TK Cross: Tenkan crosses Kijun
    tk_bullish = tenkan_aligned > kijun_aligned
    tk_bearish = tenkan_aligned < kijun_aligned
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Ichimoku (52), EMA50 (50), volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        trend_val = ema50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(senkou_a_val) or 
            np.isnan(senkou_b_val) or np.isnan(trend_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Kumo relationship
        price_above_kumo = close_val > max(senkou_a_val, senkou_b_val)
        price_below_kumo = close_val < min(senkou_a_val, senkou_b_val)
        
        # Entry conditions: Multiple confluence factors
        # Long: Price above Kumo + TK bullish + Kumo bullish (A>B) + Uptrend + Volume
        long_condition = (price_above_kumo and tk_bullish[i] and kumo_bullish[i] and 
                         is_uptrend and vol_conf)
        
        # Short: Price below Kumo + TK bearish + Kumo bearish (A<B) + Downtrend + Volume
        short_condition = (price_below_kumo and tk_bearish[i] and kumo_bearish[i] and 
                          is_downtrend and vol_conf)
        
        # Exit conditions: Kumo twist reversal or TK cross reversal
        long_exit = (position == 1 and 
                    (price_below_kumo or not tk_bullish[i] or not kumo_bullish[i] or not is_uptrend))
        short_exit = (position == -1 and 
                     (price_above_kumo or not tk_bearish[i] or not kumo_bearish[i] or not is_downtrend))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_KumoTwist_1dTrend_WeeklyVolume_v1"
timeframe = "6h"
leverage = 1.0