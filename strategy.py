#!/usr/bin/env python3
# 4h_donchian_breakout_volume_regime_v1
# Hypothesis: 4h Donchian(20) breakouts with volume confirmation and 12h/1d regime filter.
# Long: price breaks above Donchian(20) upper band with volume > 1.5x 20-period average AND 12h close > 12h EMA50 (bullish regime)
# Short: price breaks below Donchian(20) lower band with volume > 1.5x 20-period average AND 12h close < 12h EMA50 (bearish regime)
# Exit: price crosses Donchian(20) midline or ATR-based stoploss (2x ATR)
# Uses 4h primary timeframe with 12h HTF for EMA regime filter.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 12h data for EMA regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h
    ema_50_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(df_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 / (50 + 1)) + (ema_50_12h[i-1] * (49 / (50 + 1)))
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(mid[i]) or np.isnan(atr[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        regime_bullish = close[i] > ema_50_12h_aligned[i]
        regime_bearish = close[i] < ema_50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses midline OR stoploss hit (2x ATR below entry)
            if price <= mid[i] or price <= entry_price - 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses midline OR stoploss hit (2x ATR above entry)
            if price >= mid[i] or price >= entry_price + 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above upper band with volume AND bullish regime
            if price > upper[i] and vol_r > 1.5 and regime_bullish:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume AND bearish regime
            elif price < lower[i] and vol_r > 1.5 and regime_bearish:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals