#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Trend_1dChopRegime_v1
Hypothesis: 12h Williams Alligator (SMMA) trend filter with 1d choppiness regime (CHOP > 61.8 = range, < 38.2 = trend) and volume confirmation (>1.5x 20-bar MA). 
Long when price > Alligator Jaw (teeth) AND CHOP < 38.2 (trending) AND volume confirmed. 
Short when price < Alligator Jaw (teeth) AND CHOP < 38.2 (trending) AND volume confirmed. 
In choppy regimes (CHOP > 61.8), fade extremes: short near Donchian(20) upper, long near Donchian(20) lower. 
Discrete sizing (0.25) and ATR-based stoploss (2.5x) to limit churn. Target: 50-150 total trades over 4 years on 12h timeframe.
Uses 1w HTF for trend regime (EMA50) to avoid counter-trend whipsaws in bear markets.
Works in bull (trend follow with Alligator) and bear (mean revert in chop, trend follow when trending).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend regime, 1d for chop regime)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for trend regime filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Williams Alligator (SMMA: 13, 8, 5) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2.0  # Typical price for Alligator
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    alligator_jaw = smma(median_1d, 13)  # Blue line
    alligator_teeth = smma(median_1d, 8)   # Red line
    alligator_lips = smma(median_1d, 5)    # Green line
    
    # Align Alligator lines to LTF (12h)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, alligator_jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, alligator_teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, alligator_lips)
    
    # === 1d Choppiness Index (CHOP) ===
    def choppiness_index(high, low, close, period=14):
        """Choppiness Index: higher = more choppy, lower = more trending"""
        if len(close) < period:
            return np.full_like(close, np.nan, dtype=float)
        
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Wilder's ATR (SMMA of TR)
        atr_period = np.zeros_like(close)
        if len(tr) >= period:
            atr_period[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr_period[i] = (atr_period[i-1] * (period-1) + tr[i]) / period
        atr = atr_period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i >= period-1:
                atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(len(close)):
            if i >= period-1:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # CHOP = 100 * log10(atr_sum / (highest_high - lowest_low)) / log10(period)
        range_hl = highest_high - lowest_low
        # Avoid division by zero
        range_hl = np.where(range_hl == 0, 1e-10, range_hl)
        chop = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        return chop
    
    chop_values = choppiness_index(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # === 1d Donchian(20) for choppy regime mean reversion ===
    def donchian_channels(high, low, period=20):
        """Donchian Channels: upper = max(high, period), lower = min(low, period)"""
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(low, np.nan, dtype=float)
        for i in range(len(high)):
            if i >= period-1:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = donchian_channels(high_1d, low_1d, 20)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    
    # === 12h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(close)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # === 12h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma_20[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_ma_20[i] = (vol_ma_20[i-1] * 19 + volume[i]) / 20
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr[i]
        donch_upper_val = donch_upper_aligned[i]
        donch_lower_val = donch_lower_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime from 1w EMA50
        is_bull_1w = price > ema_50_1w_val
        is_bear_1w = price < ema_50_1w_val
        
        # Chop regime: >61.8 = choppy (range), <38.2 = trending
        is_choppy = chop_val > 61.8
        is_trending = chop_val < 38.2
        # Neutral zone (38.2-61.8) - default to trend following
        
        if position == 0:
            if is_trending:
                # Trending regime: follow Alligator (price > teeth = bullish, price < teeth = bearish)
                if price > teeth_val and vol_conf:
                    # Bullish alignment
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
                elif price < teeth_val and vol_conf:
                    # Bearish alignment
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
            elif is_choppy:
                # Choppy regime: mean reversion at Donchian extremes
                if price <= donch_lower_val and vol_conf:
                    # Near lower band -> long (mean reversion up)
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
                elif price >= donch_upper_val and vol_conf:
                    # Near upper band -> short (mean reversion down)
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
            # Neutral regime (38.2 <= CHOP <= 61.8): default to trend following like trending
            else:
                if price > teeth_val and vol_conf:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
                elif price < teeth_val and vol_conf:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price crosses below Alligator lips in trending regime
                elif is_trending and price < lips_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price moves back from lower Donchian in choppy regime
                elif is_choppy and price > donch_lower_val + 0.1 * (donch_upper_val - donch_lower_val):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price crosses above Alligator lips in trending regime
                elif is_trending and price > lips_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price moves back from upper Donchian in choppy regime
                elif is_choppy and price < donch_upper_val - 0.1 * (donch_upper_val - donch_lower_val):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Trend_1dChopRegime_v1"
timeframe = "12h"
leverage = 1.0