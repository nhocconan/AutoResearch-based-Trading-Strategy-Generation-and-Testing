#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeConfirm_ATRStop_v1
Hypothesis: Donchian(20) breakout on 12h with 1w EMA50 trend filter, volume spike confirmation (>2.0), and ATR(14) trailing stop (2.5x). Designed for very low trade frequency (~15-30/year) to minimize fee drag and improve generalization across bull/bear markets. Uses 12h primary timeframe with 1w HTF for trend and volume context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w trend filter: 50-period EMA ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1w volume average (20-period) for spike detection ===
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w[np.isnan(vol_ma_1w)] = 1.0  # avoid division by zero
    vol_ratio_1w = volume_1w / vol_ma_1w
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    
    # === ATR for dynamic stoploss (14-period on 12h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Donchian channels (20-period on 12h) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ratio_1w_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1w = ema_50_1w_aligned[i]
        vol_spike = vol_ratio_1w_aligned[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume spike > 2.0 + price above 1w EMA50
            if price_close > upper_donchian and vol_spike > 2.0 and price_close > trend_1w:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below lower Donchian + volume spike > 2.0 + price below 1w EMA50
            elif price_close < lower_donchian and vol_spike > 2.0 and price_close < trend_1w:
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

name = "12h_Donchian20_Breakout_1wTrend_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0