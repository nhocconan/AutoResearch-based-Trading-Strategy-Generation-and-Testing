#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_1dTrend_VolumeConfirm_v2
Hypothesis: Weekly Camarilla pivot breakouts with 1d EMA34 trend filter and volume spike confirmation. 
Weekly pivots provide strong structural levels; breakouts with volume and 1d trend alignment capture 
momentum moves. Designed for low frequency (~20-40 trades/year) to minimize fee drag. 
Uses 6h primary timeframe with 1d and 1w HTF for context. Works in both bull (breakout continuation) 
and bear (mean reversion at extremes) via volume confirmation and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 40 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d EMA34 trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Weekly Camarilla pivot levels (using prior week OHLC) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # Formula based on prior week's range
    camarilla_h4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w) / 4
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w) / 4
    camarilla_h2 = close_1w + 1.1 * (high_1w - low_1w) / 6
    camarilla_l2 = close_1w - 1.1 * (high_1w - low_1w) / 6
    camarilla_h1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    camarilla_l1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # === ATR for dynamic stoploss (14-period on 6h) ===
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
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above weekly H4 + volume spike > 2.0 + price above 1d EMA34
            if price_close > h4 and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below weekly L4 + volume spike > 2.0 + price below 1d EMA34
            elif price_close < l4 and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 2.0 * ATR below highest since entry
                if price_close < highest_since_entry - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.0 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_1dTrend_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0