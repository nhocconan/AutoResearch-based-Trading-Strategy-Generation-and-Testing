#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ChopRegime_v1
Hypothesis: Combines Donchian(20) breakout with 12h EMA50 trend filter, volume spike confirmation (>2.0x 20-period average), and Choppiness Index regime filter (CHOP < 50 for trending markets). Designed for low trade frequency (~20-40/year) to minimize fee drag and improve generalization across bull/bear markets. Uses 4h primary timeframe with 12h HTF for trend and volume context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h trend filter: 50-period EMA ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 12h volume average (20-period) for spike detection ===
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
    
    # === Donchian channels (20-period on 4h) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Choppiness Index (14-period on 4h) ===
    atr_14_chop = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_14 - lowest_14
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(atr_14_chop / chop_denom_safe) / np.log10(14)
    # Handle edge cases where chop_denom is zero or very small
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_12h = ema_50_12h_aligned[i]
        vol_spike = vol_ratio_12h_aligned[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        atr_val = atr_14[i]
        chop_val = chop[i]
        
        # Only trade in trending markets (CHOP < 50)
        if chop_val >= 50:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume spike > 2.0 + price above 12h EMA50
            if price_close > upper_donchian and vol_spike > 2.0 and price_close > trend_12h:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below lower Donchian + volume spike > 2.0 + price below 12h EMA50
            elif price_close < lower_donchian and vol_spike > 2.0 and price_close < trend_12h:
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

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0