#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_OrderBlock_BullBear_Signal_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Order Block Detection (Bullish/Bearish) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Bullish Order Block: Down candle followed by up candle with volume
    bullish_ob = (close_1d[1:] < high_1d[:-1]) & (close_1d[:-1] < open_1d) & (volume_1d[1:] > volume_1d[:-1] * 1.5)
    bearish_ob = (close_1d[1:] > low_1d[:-1]) & (close_1d[:-1] > open_1d) & (volume_1d[1:] > volume_1d[:-1] * 1.5)
    
    # Get open prices for OB detection
    open_1d = df_1d['open'].values
    bullish_ob = (close_1d[1:] < open_1d[:-1]) & (close_1d[:-1] > open_1d[:-1]) & (volume_1d[1:] > volume_1d[:-1] * 1.5)
    bearish_ob = (close_1d[1:] > open_1d[:-1]) & (close_1d[:-1] < open_1d[:-1]) & (volume_1d[1:] > volume_1d[:-1] * 1.5)
    
    # Create signal arrays (previous candle properties)
    bullish_signal = np.zeros(len(close_1d))
    bearish_signal = np.zeros(len(close_1d))
    bullish_signal[1:] = bullish_ob
    bearish_signal[1:] = bearish_ob
    
    # Align to 1h timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_signal)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_signal)
    
    # === 1h Entry Filters ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    # Momentum filter: price > 20-period EMA for long, < 20-period EMA for short
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any filter is not ready
        if np.isnan(vol_ma20[i]) or np.isnan(ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish_signal_val = bullish_aligned[i]
        bearish_signal_val = bearish_aligned[i]
        
        if position == 0:
            # Long: Bullish order block + volume + price above EMA20
            if bullish_signal_val and vol_filter[i] and close[i] > ema20[i]:
                signals[i] = 0.20
                position = 1
            # Short: Bearish order block + volume + price below EMA20
            elif bearish_signal_val and vol_filter[i] and close[i] < ema20[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below EMA20 or bearish signal appears
            if close[i] < ema20[i] or bearish_signal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price breaks above EMA20 or bullish signal appears
            if close[i] > ema20[i] or bullish_signal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals