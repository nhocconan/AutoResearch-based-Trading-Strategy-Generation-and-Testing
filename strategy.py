#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1d EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses the opposite Donchian band (mean reversion) OR ATR-based stoploss hit (2.5 * ATR).
Uses 1d HTF for trend filter to improve robustness in both bull and bear markets.
Target: 20-40 trades/year per symbol to minimize fee drag while maintaining edge.
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
    
    # Get 4h data for Donchian calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    def calculate_donchian(high_arr, low_arr, window):
        """Donchian channels: upper = max(high, window), lower = min(low, window)"""
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    # Calculate ATR (14-period) for stoploss on 4h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    # Get Donchian channels (20-period)
    upper, lower = calculate_donchian(high_4h, low_4h, 20)
    
    # Get 1d data for EMA50 trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss on 4h
    atr = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Align all indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above upper band + price > 1d EMA50 + volume spike
            if price > upper_val and price > ema50 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Breakout below lower band + price < 1d EMA50 + volume spike
            elif price < lower_val and price < ema50 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price crosses below lower band (mean reversion)
            if price < lower_val:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR below entry)
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Price crosses above upper band (mean reversion)
            if price > upper_val:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR above entry)
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0