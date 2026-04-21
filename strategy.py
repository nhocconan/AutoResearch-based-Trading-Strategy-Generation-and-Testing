# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_1d_1w_AdaptiveBreakout_VolumeRegime
Hypothesis: Combines 6h price breakouts from Donchian channels with 1d volume confirmation and 1w trend filter to capture strong momentum moves while avoiding false signals in choppy markets. Uses adaptive position sizing based on volatility regime and discrete sizing to minimize fee churn. Designed for low trade frequency (target: 15-30/year) to thrive in both bull and bear markets by focusing on high-probability breakouts with institutional volume backing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d and 1w data once for filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = np.zeros_like(vol_1d)
    for i in range(len(vol_1d)):
        if i >= 20:
            vol_avg_1d[i] = np.mean(vol_1d[i-20:i])
        else:
            vol_avg_1d[i] = np.mean(vol_1d[:i+1]) if i > 0 else vol_1d[i]
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.zeros_like(close_1w)
    if len(close_1w) >= 50:
        ema50_1w[0] = close_1w[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    else:
        ema50_1w[:] = close_1w[0] if len(close_1w) > 0 else 0
    
    # Align 1d volume average and 1w EMA50 to 6h timeframe
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        else:
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
    
    # ATR for volatility regime and stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    # Volatility regime: normalize ATR by its 50-period average
    atr_ma = np.zeros_like(atr)
    for i in range(len(atr)):
        if i >= 50:
            atr_ma[i] = np.mean(atr[i-50:i])
        else:
            atr_ma[i] = np.mean(atr[:i+1]) if i > 0 else atr[i]
    vol_regime = atr / (atr_ma + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_avg_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_avg = vol_avg_1d_aligned[i]
        ema50 = ema50_1w_aligned[i]
        atr_val = atr[i]
        vol_reg = vol_regime[i]
        
        # Stoploss: 2.0 * ATR from entry
        if position == 1 and price < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and 1w uptrend
            vol_confirm = volume[i] > 1.5 * vol_avg
            trend_filter = price > ema50
            if price > donchian_high[i] and vol_confirm and trend_filter:
                # Adaptive sizing: reduce size in high volatility
                base_size = 0.25
                vol_factor = np.clip(1.0 / vol_reg, 0.5, 2.0)  # Invert vol regime
                size = base_size * vol_factor
                size = min(max(size, 0.1), 0.4)  # Clamp to reasonable range
                signals[i] = size
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low with volume confirmation and 1w downtrend
            elif price < donchian_low[i] and vol_confirm and (price < ema50):
                base_size = 0.25
                vol_factor = np.clip(1.0 / vol_reg, 0.5, 2.0)
                size = base_size * vol_factor
                size = min(max(size, 0.1), 0.4)
                signals[i] = -size
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below Donchian low or loses 1w uptrend
            if price < donchian_low[i] or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = vol_avg_1d_aligned[i]  # Hold current size
        
        elif position == -1:
            # Short exit: price rises above Donchian high or loses 1w downtrend
            if price > donchian_high[i] or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -vol_avg_1d_aligned[i]  # Hold current size (negative)
    
    return signals

name = "6h_1d_1w_AdaptiveBreakout_VolumeRegime"
timeframe = "6h"
leverage = 1.0