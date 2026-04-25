#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout + Weekly Kumo Twist + Volume Confirmation
Hypothesis: Ichimoku cloud acts as dynamic support/resistance; Kumo twist (senkou span A/B cross) signals trend acceleration.
Breakouts in direction of twist with volume confirmation capture strong moves. Works in bull/bear via cloud filtering.
Target: 12-30 trades/year on 6h.
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
    
    # Load 1d data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Kumo twist: Senkou A crossing above/below Senkou Bullish twist: Senkou A > Senkou B (after being below)
    # Bearish twist: Senkou A < Senkou B (after being above)
    # We use completed twist: current Senkou A > Senkou B AND previous Senkou A <= Senkou B (bullish)
    # Or current Senkou A < Senkou B AND previous Senkou A >= Senkou B (bearish)
    senkou_a_vals = senkou_a.values
    senkou_b_vals = senkou_b.values
    bullish_twist = (senkou_a_vals > senkou_b_vals) & (np.roll(senkou_a_vals, 1) <= np.roll(senkou_b_vals, 1))
    bearish_twist = (senkou_a_vals < senkou_b_vals) & (np.roll(senkou_a_vals, 1) >= np.roll(senkou_b_vals, 1))
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_vals)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_vals)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    # Load 1w data for weekly context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 10:
        # Weekly EMA21 for higher timeframe trend
        close_1w = df_1w['close'].values
        ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
        ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
        weekly_uptrend = close > ema_21_1w_aligned
        weekly_downtrend = close < ema_21_1w_aligned
    else:
        weekly_uptrend = np.ones(n, dtype=bool)
        weekly_downtrend = np.ones(n, dtype=bool)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(52, 26, 9, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(bullish_twist_aligned[i]) or np.isnan(bearish_twist_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Ichimoku signals
        price_above_cloud = curr_close > max(senkou_a_aligned[i], senkou_b_aligned[i])
        price_below_cloud = curr_close < min(senkou_a_aligned[i], senkou_b_aligned[i])
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Weekly trend alignment
        w_uptrend = weekly_uptrend[i]
        w_downtrend = weekly_downtrend[i]
        
        # Kumo twist signals (completed twists)
        bullish_twist_signal = bullish_twist_aligned[i] > 0.5
        bearish_twist_signal = bearish_twist_aligned[i] > 0.5
        
        if position == 0:
            # Look for entry signals
            # Long: price above cloud + TK bullish + bullish twist + weekly uptrend + volume spike
            long_entry = (price_above_cloud and tk_bullish and bullish_twist_signal and 
                         w_uptrend and vol_spike)
            # Short: price below cloud + TK bearish + bearish twist + weekly downtrend + volume spike
            short_entry = (price_below_cloud and tk_bearish and bearish_twist_signal and 
                          w_downtrend and vol_spike)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below cloud OR TK bearish crossover
            if price_below_cloud or tk_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above cloud OR TK bullish crossover
            if price_above_cloud or tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyKumoTwist_Trend_Filter"
timeframe = "6h"
leverage = 1.0