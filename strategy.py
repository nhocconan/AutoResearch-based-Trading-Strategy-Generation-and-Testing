#!/usr/bin/env python3
"""
6h_Adaptive_Regime_Breakout_v1
Hypothesis: 6h timeframe with adaptive regime detection (choppiness index) and Donchian breakout. In trending regimes (CHOP < 38.2), breakout continuation; in ranging regimes (CHOP > 61.8), mean reversion at Donchian mid-point. Uses 12h trend filter for institutional alignment and volume confirmation to reduce false signals. Designed for low trade frequency (12-30/year) to work in both bull and bear markets by adapting to regime.
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
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Choppiness Index (14-period) for regime detection ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(atr_sum / hl_range) / np.log10(14)
    chop = np.where(hl_range == 0, 50, chop)  # neutral when no range
    
    # === Donchian Channel (20-period) ===
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # === Volume confirmation (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.where(vol_ma == 0, 0, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop[i]) or
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(dc_middle[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_12h = ema_50_12h_aligned[i]
        chop_val = chop[i]
        vol_spike = vol_ratio[i]
        upper = dc_upper[i]
        lower = dc_lower[i]
        middle = dc_middle[i]
        
        # Regime filters
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        is_neutral = ~(is_trending | is_ranging)
        
        if position == 0:
            # Long entry conditions
            long_breakout = price_close > upper and vol_spike > 1.5
            long_pullback = price_close > middle and price_close < upper * 1.02 and price_close > lower and vol_spike > 1.2
            
            # Short entry conditions
            short_breakout = price_close < lower and vol_spike > 1.5
            short_pullback = price_close < middle and price_close > lower * 0.98 and price_close < upper and vol_spike > 1.2
            
            # Trend regime: follow breakout with 12h trend filter
            if is_trending:
                if long_breakout and price_close > trend_12h:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout and price_close < trend_12h:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime: mean reversion at extremes
            elif is_ranging:
                if price_close < lower * 1.01 and vol_spike > 1.3:  # near lower band
                    signals[i] = 0.25
                    position = 1
                elif price_close > upper * 0.99 and vol_spike > 1.3:  # near upper band
                    signals[i] = -0.25
                    position = -1
            # Neutral regime: wait for clearer signal
            else:
                if long_breakout and price_close > trend_12h:
                    signals[i] = 0.20
                    position = 1
                elif short_breakout and price_close < trend_12h:
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:  # Long position
                # Stop loss: 2.0 * ATR below entry (simplified as 2% for now)
                if price_close < middle:  # exit at middle line
                    signals[i] = 0.0
                    position = 0
                # Trailing profit target: exit if price drops 1.5% from high
                elif price_high > 0 and price_close < price_high * 0.985:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Short position
                # Stop loss: 2.0 * ATR above entry
                if price_close > middle:  # exit at middle line
                    signals[i] = 0.0
                    position = 0
                # Trailing profit target: exit if price rises 1.5% from low
                elif price_low > 0 and price_close > price_low * 1.015:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Adaptive_Regime_Breakout_v1"
timeframe = "6h"
leverage = 1.0