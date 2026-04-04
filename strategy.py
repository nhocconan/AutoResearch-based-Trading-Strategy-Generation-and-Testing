#!/usr/bin/env python3
"""
Experiment #4815: 6h Ichimoku Cloud Breakout + 1w Trend Filter + Volume Spike
HYPOTHESIS: On 6h timeframe, Ichimoku cloud breaks in direction of 1w Kumo twist with volume confirmation (>1.5x average) capture strong momentum in both bull and bear markets. The Kumo twist (Senkou Span A/B cross) acts as a reliable trend reversal filter. Target: 15-30 trades/year to minimize fee drag while maintaining statistical significance. Works in bull markets (breaks above cloud with bullish twist) and bear markets (breaks below cloud with bearish twist).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4815_6h_ichimoku_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for Kumo twist trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Ichimoku components for Kumo twist ===
    if len(df_1w) >= 52:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        senkou_a = ((tenkan_sen + kijun_sen) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
        senkou_b = ((period52_high + period52_low) / 2)
        
        # Kumo twist: Senkou Span A crossing above/below Senkou Span B
        # Bullish twist: Senkou A > Senkou B (uptrend)
        # Bearish twist: Senkou A < Senkou B (downtrend)
        kumo_twist_bullish = senkou_a > senkou_b
        kumo_twist_bearish = senkou_a < senkou_b
    else:
        kumo_twist_bullish = np.full(len(df_1w), False)
        kumo_twist_bearish = np.full(len(df_1w), False)
    
    # Align HTF Kumo twist to 6h timeframe
    if len(kumo_twist_bullish) > 0:
        kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1w, kumo_twist_bullish.astype(float))
        kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1w, kumo_twist_bearish.astype(float))
        kumo_twist_bullish_aligned = kumo_twist_bullish_aligned > 0.5
        kumo_twist_bearish_aligned = kumo_twist_bearish_aligned > 0.5
    else:
        kumo_twist_bullish_aligned = np.full(n, False)
        kumo_twist_bearish_aligned = np.full(n, False)
    
    # === 6h Indicators: Ichimoku Cloud ===
    if len(high) >= 52:
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
        tenkan_sen_6h = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
        kijun_sen_6h = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        senkou_a_6h = ((tenkan_sen_6h + kijun_sen_6h) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high_6h = pd.Series(high).rolling(window=52, min_periods=52).max().values
        period52_low_6h = pd.Series(low).rolling(window=52, min_periods=52).min().values
        senkou_b_6h = ((period52_high_6h + period52_low_6h) / 2)
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
        cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    else:
        tenkan_sen_6h = np.full(n, np.nan)
        kijun_sen_6h = np.full(n, np.nan)
        senkou_a_6h = np.full(n, np.nan)
        senkou_b_6h = np.full(n, np.nan)
        cloud_top = np.full(n, np.nan)
        cloud_bottom = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(52, 20, 14)  # Ichimoku, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Ichimoku cloud breakout conditions with Kumo twist alignment
        bullish_breakout = (price > cloud_top[i]) and kumo_twist_bullish_aligned[i] and vol_confirm
        bearish_breakout = (price < cloud_bottom[i]) and kumo_twist_bearish_aligned[i] and vol_confirm
        
        # Final entry conditions
        if bullish_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif bearish_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals