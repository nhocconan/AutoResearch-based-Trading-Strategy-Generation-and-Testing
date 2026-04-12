#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_elliott_wave_oscillator_v1
# Uses 6h Elliott Wave Oscillator (34-period SMA minus 5-period SMA) with 1d trend filter.
# In trending markets (1d ADX > 25), enters long when EWO crosses above zero,
# short when EWO crosses below zero. In range-bound markets (ADX <= 25),
# fades extreme EWO readings (> |30|) with mean reversion.
# Volume confirmation requires current volume > 1.3x 20-period average.
# Target: 20-40 trades/year per symbol to minimize fee drag while capturing trend and reversal edges.

name = "6h_1d_elliott_wave_oscillator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(atr_1d != 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di = np.where(atr_1d != 0, 100 * minus_dm_smooth / atr_1d, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = wilders_smooth(dx, 14)
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h Elliott Wave Oscillator: 34-period SMA minus 5-period SMA
    def sma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if i < period - 1:
                continue
            result[i] = np.mean(arr[i-period+1:i+1])
        return result
    
    sma_5 = sma(close, 5)
    sma_34 = sma(close, 34)
    ewo = sma_5 - sma_34  # Positive = bullish momentum, Negative = bearish momentum
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if required data not ready
        if np.isnan(ewo[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_confirm[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: trending if ADX > 25, ranging if ADX <= 25
        trending = adx_1d_aligned[i] > 25
        
        if trending:
            # Trend following mode: EWO crossovers
            if ewo[i] > 0 and ewo[i-1] <= 0 and position != 1:  # Bullish crossover
                position = 1
                signals[i] = 0.25
            elif ewo[i] < 0 and ewo[i-1] >= 0 and position != -1:  # Bearish crossover
                position = -1
                signals[i] = -0.25
            # Exit on opposite crossover
            elif ewo[i] < 0 and ewo[i-1] >= 0 and position == 1:  # Bearish cross while long
                position = 0
                signals[i] = 0.0
            elif ewo[i] > 0 and ewo[i-1] <= 0 and position == -1:  # Bullish cross while short
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Mean reversion mode: fade extreme EWO readings
            if ewo[i] < -30 and position != 1:  # Oversold, go long
                position = 1
                signals[i] = 0.25
            elif ewo[i] > 30 and position != -1:  # Overbought, go short
                position = -1
                signals[i] = -0.25
            # Exit when EWO returns toward zero
            elif abs(ewo[i]) < 10 and position != 0:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        
        # Apply volume confirmation: if no volume confirmation, hold or flatten
        if not vol_confirm[i]:
            if position == 1:
                signals[i] = 0.25  # hold long
            elif position == -1:
                signals[i] = -0.25  # hold short
            else:
                signals[i] = 0.0  # stay flat
    
    return signals