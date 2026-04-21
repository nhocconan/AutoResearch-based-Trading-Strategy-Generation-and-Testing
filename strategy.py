#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
Designed for low trade frequency (~20-30/year) to minimize fee drag. Uses 4h primary timeframe with 12h HTF for trend/volume context.
Camarilla levels derived from prior 1d OHLC, providing institutional pivot points that work in both bull/bear markets via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load 1d data ONCE for Camarilla calculation (prior day OHLC) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate Camarilla levels from prior 1d candle ===
    # Camarilla: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's OHLC (no look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First value will be NaN due to roll, handled by min_periods later
    
    range_1d = high_1d_prev - low_1d_prev
    # Camarilla R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    camarilla_multiplier = 1.1 / 12
    r1_level = close_1d_prev + (range_1d * camarilla_multiplier)
    s1_level = close_1d_prev - (range_1d * camarilla_multiplier)
    
    # Align Camarilla levels to 4h timeframe (already uses prior day's data)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # === Load 12h HTF data ONCE for trend and volume filters ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h volume average (20-period) for spike detection
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h[np.isnan(vol_ma_12h)] = 1.0  # avoid division by zero
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === ATR for dynamic stoploss (14-period on 4h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        trend_12h = ema_50_12h_aligned[i]
        vol_spike = vol_ratio_12h_aligned[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 12h EMA50
            if price_close > r1 and vol_spike > 2.0 and price_close > trend_12h:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below S1 + volume spike > 2.0 + price below 12h EMA50
            elif price_close < s1 and vol_spike > 2.0 and price_close < trend_12h:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 2.5 * ATR below highest since entry
                if price_close < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.5 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0