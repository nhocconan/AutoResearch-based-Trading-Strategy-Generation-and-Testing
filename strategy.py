#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud with Kumo twist + Tenkan-Kijun cross + volume confirmation.
# Ichimoku Cloud acts as dynamic support/resistance. Kumo twist (Senkou A/B cross) signals trend change.
# Tenkan-Kijun cross provides entry signals in direction of Kumo twist and price relative to cloud.
# Volume confirms momentum. Works in bull/bear by adapting to cloud color and twist.
# Target: 15-30 trades/year per symbol (45-120 total over 4 years).
name = "6h_Ichimoku_KumoTwist_TK_Cross_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26  # Kumo cloud displacement
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Chikou Span (Lagging Span): not needed for this strategy
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Kumo twist: Senkou A/B cross (trend change signal)
    # We use the previous value to detect cross (avoiding look-ahead)
    senkou_a_prev = senkou_span_a_aligned[:-1]
    senkou_b_prev = senkou_span_b_aligned[:-1]
    senkou_a_curr = senkou_span_a_aligned[1:]
    senkou_b_curr = senkou_span_b_aligned[1:]
    
    # Bullish twist: Senkou A crosses above Senkou B
    bullish_twist = np.zeros(n, dtype=bool)
    bearish_twist = np.zeros(n, dtype=bool)
    bullish_twist[1:] = (senkou_a_prev < senkou_b_prev) & (senkou_a_curr > senkou_b_curr)
    bearish_twist[1:] = (senkou_a_prev > senkou_b_prev) & (senkou_a_curr < senkou_b_curr)
    
    # Kumo cloud color: Senkou A > Senkou B = bullish cloud
    bullish_cloud = senkou_span_a_aligned > senkou_span_b_aligned
    bearish_cloud = senkou_span_a_aligned < senkou_span_b_aligned
    
    # Tenkan-Kijun cross
    tk_cross_up = np.zeros(n, dtype=bool)
    tk_cross_down = np.zeros(n, dtype=bool)
    tk_cross_up[1:] = (tenkan_sen_aligned[:-1] < kijun_sen_aligned[:-1]) & (tenkan_sen_aligned[1:] > kijun_sen_aligned[1:])
    tk_cross_down[1:] = (tenkan_sen_aligned[:-1] > kijun_sen_aligned[:-1]) & (tenkan_sen_aligned[1:] < kijun_sen_aligned[1:])
    
    # Price relative to cloud
    price_above_cloud = (close > senkou_span_a_aligned) & (close > senkou_span_b_aligned)
    price_below_cloud = (close < senkou_span_a_aligned) & (close < senkou_span_b_aligned)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52 + displacement, 20)  # Ensure Ichimoku and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: bullish Kumo twist OR (bullish cloud AND Tenkan-Kijun cross up) with price above cloud and volume
            if ((bullish_twist[i] or 
                 (bullish_cloud[i] and tk_cross_up[i] and price_above_cloud[i])) and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish Kumo twist OR (bearish cloud AND Tenkan-Kijun cross down) with price below cloud and volume
            elif ((bearish_twist[i] or 
                   (bearish_cloud[i] and tk_cross_down[i] and price_below_cloud[i])) and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish Kumo twist OR price closes below cloud OR Tenkan-Kijun cross down
            if bearish_twist[i] or (close[i] < senkou_span_a_aligned[i] and close[i] < senkou_span_b_aligned[i]) or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish Kumo twist OR price closes above cloud OR Tenkan-Kijun cross up
            if bullish_twist[i] or (close[i] > senkou_span_a_aligned[i] and close[i] > senkou_span_b_aligned[i]) or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals