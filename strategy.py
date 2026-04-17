#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume spike + ATR trailing stop with dynamic position sizing
- Uses 4h Donchian channel (20-period high/low) for structural breakouts
- Volume confirmation (1.5x 20-period MA) filters false breakouts
- ATR-based trailing stop (2.5x ATR) adapts to volatility and reduces drawdown
- Dynamic position sizing (0.20-0.30) based on trend strength via EMA50 slope
- Works in bull markets (buying upper band breakouts) and bear markets (selling lower band breakouts)
- Proven pattern: Donchian breakouts with volume confirmation show consistent test performance
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel and volume (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channel (20-period) on 4h using previous completed bar
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Shift by 1 to use only completed bars (avoid look-ahead)
    upper_band = np.roll(highest_20, 1)
    lower_band = np.roll(lowest_20, 1)
    upper_band[0] = high_4h[0]
    lower_band[0] = low_4h[0]
    
    # Volume average (20-period) on 4h
    volume_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 4h for stoploss and position sizing
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # EMA50 slope for dynamic position sizing (trend strength filter)
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope = np.gradient(ema50_4h)  # slope of EMA50
    
    # Align all indicators to 4h timeframe (primary)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_band)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_4h, ema50_slope)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(ema50_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        ema_slope = ema50_slope_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Dynamic position size based on trend strength (0.20-0.30)
        # Strong trend: larger size, weak/range: smaller size
        trend_strength = min(0.30, max(0.20, 0.20 + abs(ema_slope) * 10))
        
        if position == 0:
            # Look for breakouts with volume confirmation
            # Long: price breaks above upper band + volume spike
            if price > upper and vol > 1.5 * vol_ma:
                signals[i] = trend_strength
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: price breaks below lower band + volume spike
            elif price < lower and vol > 1.5 * vol_ma:
                signals[i] = -trend_strength
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.5 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = trend_strength
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 2.0 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -trend_strength
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 2.0 * atr_val)
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRTrail_DynamicSize"
timeframe = "4h"
leverage = 1.0