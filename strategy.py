#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Confluence_v2
Hypothesis: On 6h timeframe, price breaking above/below 20-period Donchian channels with confluence from weekly pivot levels (R1/S1) and volume confirmation captures institutional breakouts while filtering false signals. Weekly pivot provides longer-term structure, Donchian gives breakout timing, volume confirms participation. Designed for 6h TF to target 50-150 total trades over 4 years (12-37/year) to minimize fee drag and work in both bull/bear regimes via pivot mean reversion in ranges and breakout filtering in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for daily ATR, 1w for weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    # === Daily ATR for volatility filtering and stoploss ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Weekly Pivot Points (R1, S1, PP) from prior week ===
    # Using prior week's OHLC to avoid look-ahead
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly Pivot Point calculation
    weekly_pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_r1 = 2 * weekly_pp - prev_week_low
    weekly_s1 = 2 * weekly_pp - prev_week_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === 6h Donchian Channel (20-period) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate rolling max/min for Donchian channels
    highest_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_volume[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol = volume[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        vol_avg = avg_volume[i]
        atr_val = atr_1d_aligned[i]
        weekly_pp_val = weekly_pp_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        
        # Volume spike filter: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly R1 + volume spike
            if price_high > donchian_high and price_close > weekly_r1_val and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below Donchian low + below weekly S1 + volume spike
            elif price_low < donchian_low and price_close < weekly_s1_val and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # ATR-based stoploss and pivot mean reversion exit
            if position == 1:
                # Stoploss: 2 * ATR below entry
                stop_price = entry_price - 2.0 * atr_val
                # Exit if price hits stop or mean reversion to weekly pivot
                if price_low < stop_price or price_close < weekly_pp_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Stoploss: 2 * ATR above entry
                stop_price = entry_price + 2.0 * atr_val
                # Exit if price hits stop or mean reversion to weekly pivot
                if price_high > stop_price or price_close > weekly_pp_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Confluence_v2"
timeframe = "6h"
leverage = 1.0