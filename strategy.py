#!/usr/bin/env python3
"""
exp_7259_6h_ichimoku_cloud_12h_tk_v1
Hypothesis: 6h Ichimoku with TK cross filtered by 12h cloud color for trend regime.
In bullish cloud (price > cloud): TK cross up = long, TK cross down = exit.
In bearish cloud (price < cloud): TK cross down = short, TK cross up = exit.
In neutral cloud (price inside cloud): no trades to avoid whipsaw.
Uses Ichimoku's built-in trend/filter properties to work in both bull and bear markets.
Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7259_6h_ichimoku_cloud_12h_tk_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9  # Tenkan-sen period
KJ_PERIOD = 26  # Kijun-sen period
SS_B_PERIOD = 52  # Senkou Span B period
DISPLACEMENT = 26  # Kumo cloud displacement
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8  # ~2 days (8*6h = 48h)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for Ichimoku
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Ichimoku components on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past TK_PERIOD
    highest_high_tk = pd.Series(high_12h).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max().values
    lowest_low_tk = pd.Series(low_12h).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min().values
    tenkan = (highest_high_tk + lowest_low_tk) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past KJ_PERIOD
    highest_high_kj = pd.Series(high_12h).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max().values
    lowest_low_kj = pd.Series(low_12h).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min().values
    kijun = (highest_high_kj + lowest_low_kj) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced forward
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past SS_B_PERIOD displaced
    highest_high_ss = pd.Series(high_12h).rolling(window=SS_B_PERIOD, min_periods=SS_B_PERIOD).max().values
    lowest_low_ss = pd.Series(low_12h).rolling(window=SS_B_PERIOD, min_periods=SS_B_PERIOD).min().values
    senkou_b = ((highest_high_ss + lowest_low_ss) / 2)
    
    # Align Ichimoku components to LTF (6h) with proper displacement
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period - need enough data for all Ichimoku components
    start = max(TK_PERIOD, KJ_PERIOD, SS_B_PERIOD) + DISPLACEMENT + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available (check Senkou Span B as proxy)
        if np.isnan(senkou_b_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine cloud boundaries (Senkou Span A and B)
        senkou_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        senkou_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Price relative to cloud
        price_above_cloud = close[i] > senkou_top
        price_below_cloud = close[i] < senkou_bottom
        price_in_cloud = (close[i] >= senkou_bottom) & (close[i] <= senkou_top)
        
        # TK Cross (Tenkan-sen crossing Kijun-sen)
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        # Entry logic based on cloud regime
        if position == 0:  # flat - look for new entries
            # Bullish regime: price above cloud + TK cross up = long
            if price_above_cloud and tk_cross_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Bearish regime: price below cloud + TK cross down = short
            elif price_below_cloud and tk_cross_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:  # long position - exit on TK cross down OR price drops below cloud
            if tk_cross_down or close[i] < senkou_bottom:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:  # short position - exit on TK cross up OR price rises above cloud
            if tk_cross_up or close[i] > senkou_top:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>