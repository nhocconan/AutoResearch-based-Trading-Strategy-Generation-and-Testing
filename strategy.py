#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Ichimoku (Tenkan/Kijun) cross above/below Kumo cloud with 1d EMA50 trend filter
# and volume spike to capture strong trending moves in both bull and bear markets.
# Designed for low-frequency trades (target 50-150 total) to minimize fee drag.

name = "6h_Ichimoku_1dEMA50_Volume"
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
    
    # Get 1d data for Ichimoku components and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    
    tenkan = (period9_high + period9_low) / 2
    kijun = (period26_high + period26_low) / 2
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Kumo cloud boundaries (shifted 26 periods ahead)
    senkou_a_lead = np.roll(senkou_a, 26)
    senkou_b_lead = np.roll(senkou_b, 26)
    senkou_a_lead[:26] = np.nan
    senkou_b_lead[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lead, senkou_b_lead)
    kumo_bottom = np.minimum(senkou_a_lead, senkou_b_lead)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    
    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike (2.0x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure Ichimoku has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Tenkan > Kijun (bullish cross) AND price above Kumo AND 1d uptrend AND volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > kumo_top_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Tenkan < Kijun (bearish cross) AND price below Kumo AND 1d downtrend AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < kumo_bottom_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan < Kijun OR price drops below Kumo bottom OR trend fails
            if (tenkan_aligned[i] < kijun_aligned[i] or 
                close[i] < kumo_bottom_aligned[i] or
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan > Kijun OR price rises above Kumo top OR trend fails
            if (tenkan_aligned[i] > kijun_aligned[i] or 
                close[i] > kumo_top_aligned[i] or
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals