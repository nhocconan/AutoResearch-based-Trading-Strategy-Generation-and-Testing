#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_WeeklyPivot_Filter_v1
Hypothesis: 6h Donchian(20) breakout with 1d EMA trend filter and weekly Camarilla pivot direction reduces false breakouts while capturing institutional moves. Volume confirmation ensures participation. Designed for low trade frequency (~12-37/year) to minimize fee drag on 6h timeframe. Works in both bull and bear markets by filtering breakouts with higher timeframe trend and pivot direction.
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
    if len(df_1d) < 34 or len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA on 1d ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Weekly Camarilla pivot direction (from prior weekly session) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla R4 and S4 levels (strong breakout levels)
    camarilla_r4_1w = close_1w + (high_1w - low_1w) * 1.1 / 2  # R4 = close + 1.1*range/2
    camarilla_s4_1w = close_1w - (high_1w - low_1w) * 1.1 / 2  # S4 = close - 1.1*range/2
    
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # === Donchian channel (20-period on 6h) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Donchian upper and lower bands (20-period)
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === Volume spike filter (20-period on 6h) ===
    volume_6h = prices['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_6h
    
    # === ATR for dynamic stoploss (14-period on 6h) ===
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1)) if 'close_6h' in locals() else np.abs(high_6h - np.roll(prices['close'].values, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1)) if 'close_6h' in locals() else np.abs(low_6h - np.roll(prices['close'].values, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Use close prices for Donchian (need to define close_6h)
    close_6h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r4_1w_aligned[i]) or np.isnan(camarilla_s4_1w_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ratio_6h[i]) or np.isnan(atr_14_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio_6h[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        trend_1d = ema_34_1d_aligned[i]
        r4_weekly = camarilla_r4_1w_aligned[i]
        s4_weekly = camarilla_s4_1w_aligned[i]
        atr_val = atr_14_6h[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike > 1.5 + price above 1d EMA + weekly R4 broken (bullish weekly bias)
            if price_close > upper and vol_spike > 1.5 and price_close > trend_1d and price_close > r4_weekly:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below Donchian lower + volume spike > 1.5 + price below 1d EMA + weekly S4 broken (bearish weekly bias)
            elif price_close < lower and vol_spike > 1.5 and price_close < trend_1d and price_close < s4_weekly:
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

name = "6h_Donchian20_Breakout_1dTrend_WeeklyPivot_Filter_v1"
timeframe = "6h"
leverage = 1.0