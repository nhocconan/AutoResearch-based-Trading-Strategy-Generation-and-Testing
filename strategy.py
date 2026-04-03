#!/usr/bin/env python3
"""
Experiment #095: 6h Ichimoku Cloud + ADX Trend + Volume Confirmation

HYPOTHESIS: Ichimoku cloud provides dynamic support/resistance and trend direction, 
while ADX filters for trending markets and volume confirmation ensures institutional 
participation. This combination works in both bull and bear markets by only taking 
trades in the direction of the higher timeframe trend (1d/1w) with proper filtering.
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize 
fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr = np.maximum(high_1d - low_1d, 
                       np.maximum(abs(high_1d - np.roll(close_1d, 1)), 
                                 abs(low_1d - np.roll(close_1d, 1))))
        tr[0] = high_1d[0] - low_1d[0]
        
        # Directional Movement
        up_move = high_1d - np.roll(high_1d, 1)
        down_move = np.roll(low_1d, 1) - low_1d
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Trend strength: ADX > 25 indicates trending market
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 0)
    
    # === HTF: 1w data for Ichimoku cloud (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components on 1w
    if len(df_1w) >= 52:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
        senkou_span_b = ((period52_high + period52_low) / 2)
        
        # Chikou Span (Lagging Span): Close plotted 26 periods behind
        chikou_span = np.roll(close_1w, -26)
        
        # Align Ichimoku components to LTF
        tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
        kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
        span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
        span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
        chikou_aligned = align_htf_to_ltf(prices, df_1w, chikou_span)
    else:
        tenkan_aligned = np.full(n, np.nan)
        kijun_aligned = np.full(n, np.nan)
        span_a_aligned = np.full(n, np.nan)
        span_b_aligned = np.full(n, np.nan)
        chikou_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Ichimoku Cloud Logic ---
        # Price above/below cloud
        cloud_top = np.maximum(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = np.minimum(span_a_aligned[i], span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross (Tenkan-sen crosses Kijun-sen)
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        # Chikou confirmation (price vs chikou span)
        chikou_confirm_long = close[i] > chikou_aligned[i]
        chikou_confirm_short = close[i] < chikou_aligned[i]
        
        # --- Trend Filter: Require ADX > 25 (trending market) ---
        strong_trend = adx_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic ---
        if in_position:
            # Exit when price re-enters cloud or TK cross reverses
            if position_side > 0:  # Long position
                if (close[i] <= cloud_top and close[i] >= cloud_bottom) or \
                   (tk_cross_down and not chikou_confirm_long):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if (close[i] <= cloud_top and close[i] >= cloud_bottom) or \
                   (tk_cross_up and not chikou_confirm_short):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price above cloud + TK cross up + Chikou confirmation + trend + volume
        long_condition = (
            price_above_cloud and 
            tk_cross_up and 
            chikou_confirm_long and 
            strong_trend and 
            volume_spike
        )
        
        # Short: Price below cloud + TK cross down + Chikou confirmation + trend + volume
        short_condition = (
            price_below_cloud and 
            tk_cross_down and 
            chikou_confirm_short and 
            strong_trend and 
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals