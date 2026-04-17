#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses Donchian middle band (mean reversion) or ATR-based stoploss hit.
Uses proven Donchian breakout structure with weekly trend filter to avoid counter-trend trades.
Designed for low trade frequency (7-25/year) on 1d timeframe to minimize fee drag.
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
    
    # Calculate ATR for stoploss (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Donchian channels (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        """Rolling maximum with min_periods"""
        series = pd.Series(arr)
        return series.rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        """Rolling minimum with min_periods"""
        series = pd.Series(arr)
        return series.rolling(window=window, min_periods=window).min().values
    
    donchian_upper = rolling_max(high_1d, 20)
    donchian_lower = rolling_min(low_1d, 20)
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 1w data for EMA50 trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 1d
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        duc = donchian_upper_aligned[i]
        dlc = donchian_lower_aligned[i]
        dmc = donchian_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol = volume[i]
        price = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper AND price > weekly EMA50 AND volume > 1.5x avg
            if price > duc and price > ema50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Donchian lower AND price < weekly EMA50 AND volume > 1.5x avg
            elif price < dlc and price < ema50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: price crosses Donchian middle OR ATR stoploss hit
            exit_signal = False
            if price < dmc:  # price crosses below middle band
                exit_signal = True
            elif price < entry_price - 2.0 * atr_val:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price crosses Donchian middle OR ATR stoploss hit
            exit_signal = False
            if price > dmc:  # price crosses above middle band
                exit_signal = True
            elif price > entry_price + 2.0 * atr_val:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA50_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0