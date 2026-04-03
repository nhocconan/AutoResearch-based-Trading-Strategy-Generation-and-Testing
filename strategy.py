#!/usr/bin/env python3
"""
Experiment #275: 6h Ichimoku Cloud + 1d Weekly Trend Filter + Volume Spike
HYPOTHESIS: Ichimoku TK cross (Tenkan/Kijun) signals aligned with 1d weekly trend (price above/below weekly Kumo) capture high-probability trend continuation. Volume confirmation (>2.0x average) filters false signals. Works in bull via cloud breakouts, bear via cloud rejections. Target: 75-150 total trades over 4 years (19-38/year). Uses discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_275_6h_ichimoku_1d_weekly_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly Ichimoku (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df_1d['high'].rolling(window=9, min_periods=9).max()
    period9_low = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df_1d['high'].rolling(window=26, min_periods=26).max()
    period26_low = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = df_1d['high'].rolling(window=52, min_periods=52).max()
    period52_low = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2.0).shift(52)
    
    # Chikou Span (Lagging Span): Close shifted -22 periods (not used for trend filter)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # === 6h Indicators: Ichimoku TK Cross ===
    # Tenkan-sen (6h): (9-period high + 9-period low)/2
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2.0
    
    # Kijun-sen (6h): (26-period high + 26-period low)/2
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_6h = (period26_high_6h + period26_low_6h) / 2.0
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Enough for Ichimoku calculations (52-period + 26 shift)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- 6h Ichimoku TK Cross ---
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        # --- 1d Weekly Trend Filter: Price relative to Kumo (Cloud) ---
        # Bullish trend: price above both Senkou Span A and B
        # Bearish trend: price below both Senkou Span A and B
        bullish_trend = price > span_a_aligned[i] and price > span_b_aligned[i]
        bearish_trend = price < span_a_aligned[i] and price < span_b_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on TK cross down with volume if bearish trend
                if tk_cross_down and volume_spike and bearish_trend:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on TK cross up with volume if bullish trend
                if tk_cross_up and volume_spike and bullish_trend:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + TK cross + trend alignment
        if volume_spike:
            # Long: TK cross up AND bullish trend (price above cloud)
            if tk_cross_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: TK cross down AND bearish trend (price below cloud)
            elif tk_cross_down and bearish_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals