#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeBreakout_v1
Hypothesis: Ichimoku Kumo twist (Senkou Span A/B cross) on 6h with 1d trend filter (price >/=< EMA50) and volume confirmation (2.0x average). Kumo twist signals potential trend reversal with momentum. In bull markets, long when price above Kumo and bullish twist; in bear markets, short when price below Kumo and bearish twist. Uses discrete position sizing (0.25) to limit drawdown and fee drag. Targets 50-150 total trades over 4 years.
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
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # need at least 52 for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h (no additional delay needed as they are based on completed candles)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku (52), 1d EMA (50), volume MA (20)
    start_idx = max(52, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        # Kumo twist detection: Senkou A crosses Senkou B
        # Bullish twist: Senkou A crosses above Senkou B (previous A <= previous B and current A > current B)
        # Bearish twist: Senkou A crosses below Senkou B (previous A >= previous B and current A < current B)
        if i >= 1:
            prev_senkou_a = senkou_a_aligned[i-1]
            prev_senkou_b = senkou_b_aligned[i-1]
            curr_senkou_a = senkou_a_aligned[i]
            curr_senkou_b = senkou_b_aligned[i]
            
            bullish_twist = (prev_senkou_a <= prev_senkou_b) and (curr_senkou_a > curr_senkou_b)
            bearish_twist = (prev_senkou_a >= prev_senkou_b) and (curr_senkou_a < curr_senkou_b)
        else:
            bullish_twist = False
            bearish_twist = False
        
        # Trend filter from 1d EMA50
        trend_1d_up = close_val > ema_50_1d_aligned[i]
        trend_1d_down = close_val < ema_50_1d_aligned[i]
        
        # Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_kumo = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Price relative to Kumo
        price_above_kumo = close_val > upper_kumo
        price_below_kumo = close_val < lower_kumo
        price_in_kumo = (close_val >= lower_kumo) and (close_val <= upper_kumo)
        
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: bullish Kumo twist AND price above Kumo AND 1d uptrend AND volume spike
            long_signal = bullish_twist and price_above_kumo and trend_1d_up and vol_spike
            
            # Short: bearish Kumo twist AND price below Kumo AND 1d downtrend AND volume spike
            short_signal = bearish_twist and price_below_kumo and trend_1d_down and vol_spike
            
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
            # Exit: price re-enters Kumo OR bearish Kumo twist
            if price_in_kumo or bearish_twist:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters Kumo OR bullish Kumo twist
            if price_in_kumo or bullish_twist:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0