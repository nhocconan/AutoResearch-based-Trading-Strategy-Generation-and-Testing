#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based stoploss.
Long when price breaks above Donchian upper (20) AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower (20) AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit via ATR trailing stop (signal→0 when price < highest_high - 2*ATR for longs, price > lowest_low + 2*ATR for shorts).
Uses proven price channel breakout structure with trend filter to avoid counter-trend trades.
Designed for moderate trade frequency (20-50/year) on 4h timeframe to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and ATR (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_upper = rolling_max(high_4h, 20)
    donchian_lower = rolling_min(low_4h, 20)
    
    # Calculate ATR (14) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA50 trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        atr = atr_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > upper DONCH AND close > EMA50 AND volume > 1.5x avg
            if price > upper and close[i] > ema50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_high = price
            # Short: price < lower DONCH AND close < EMA50 AND volume > 1.5x avg
            elif price < lower and close[i] < ema50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_low = price
        
        elif position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, price)
            # ATR trailing stop: exit if price < highest_high - 2*ATR
            if price < highest_high - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, price)
            # ATR trailing stop: exit if price > lowest_low + 2*ATR
            if price > lowest_low + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0