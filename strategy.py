#!/usr/bin/env python3
"""
4h Donchian Channel Breakout + Volume Spike + 1d RSI Filter
Long when price breaks above Donchian(20) high with volume spike and 1d RSI > 50,
short when breaks below Donchian(20) low with volume spike and 1d RSI < 50.
Uses 1d RSI to filter for bull/bear regime - only trade long in bull regime (RSI>50),
short in bear regime (RSI<50). Avoids whipsaws in sideways markets.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily timeframe
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder's smoothing
            for i in range(period + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Donchian channel (20-period) on 4h data
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi = rsi_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and bullish RSI (>50)
            if (price > dc_upper[i] and 
                volume_spike[i] and rsi > 50):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume spike and bearish RSI (<50)
            elif (price < dc_lower[i] and 
                  volume_spike[i] and rsi < 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses below Donchian lower (breakdown)
            if price < dc_lower[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses above Donchian upper (breakout)
            if price > dc_upper[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_1dRSIFilter"
timeframe = "4h"
leverage = 1.0