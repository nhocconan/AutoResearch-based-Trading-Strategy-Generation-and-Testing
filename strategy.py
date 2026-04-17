#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based stoploss.
Long when price breaks above Donchian upper band AND volume > 1.5x 20-period average AND price > 1d EMA50.
Short when price breaks below Donchian lower band AND volume > 1.5x 20-period average AND price < 1d EMA50.
Exit when price crosses the 1d EMA50 in opposite direction or ATR stoploss is hit.
Designed for low trade frequency (19-50/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
Uses 4h as primary timeframe with 1d HTF for trend filtering.
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
    
    # Get 4h data for Donchian calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_1d_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr = atr_14_aligned[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above upper band AND volume > 1.5x avg AND price > 1d EMA50 (bullish trend)
            if high_price > upper and vol > 1.5 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band AND volume > 1.5x avg AND price < 1d EMA50 (bearish trend)
            elif low_price < lower and vol > 1.5 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # 1. Price crosses below 1d EMA50
            if price < ema_50:
                exit_signal = True
            # 2. ATR stoploss (2.5 * ATR below entry)
            elif price <= entry_price - 2.5 * atr:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # 1. Price crosses above 1d EMA50
            if price > ema_50:
                exit_signal = True
            # 2. ATR stoploss (2.5 * ATR above entry)
            elif price >= entry_price + 2.5 * atr:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1dEMA50_ATRStop"
timeframe = "4h"
leverage = 1.0