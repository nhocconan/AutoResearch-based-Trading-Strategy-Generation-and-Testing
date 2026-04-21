#!/usr/bin/env python3
"""
1h_4hDonchian_1dTrend_VolumeConfirm_ATRStop_v1
Hypothesis: For 1h timeframe, use 4h Donchian channel breakouts filtered by 1d EMA trend and volume confirmation, with ATR-based trailing stop. This captures medium-term trends while minimizing false breakouts. Target 15-30 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 34:
        return np.zeros(n)
    
    # === 4h Donchian channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation: 4h volume > 1.5x 20-period MA ===
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / np.where(vol_ma_4h > 0, vol_ma_4h, np.nan)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # === ATR for trailing stop (14-period on 4h) ===
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio_4h_aligned[i]) or
            np.isnan(atr_14_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_4h_aligned[i]
        atr_val = atr_14_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + volume spike + price above 1d EMA
            if price_close > dh and vol_spike > 1.5 and price_close > trend_1d:
                signals[i] = 0.20
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below 4h Donchian low + volume spike + price below 1d EMA
            elif price_close < dl and vol_spike > 1.5 and price_close < trend_1d:
                signals[i] = -0.20
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
                    signals[i] = 0.20
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.5 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian_1dTrend_VolumeConfirm_ATRStop_v1"
timeframe = "1h"
leverage = 1.0