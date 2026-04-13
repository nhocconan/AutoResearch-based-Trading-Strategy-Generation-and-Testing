#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian breakout with 1w ATR regime filter and volume confirmation
    # Long: price > Donchian(20) high AND 1w ATR(14) > 1.3x 50-period average (high vol regime) AND volume > 1.5x avg
    # Short: price < Donchian(20) low AND 1w ATR(14) > 1.3x 50-period average AND volume > 1.5x avg
    # Exit: opposite Donchian breakout or volatility contraction
    # Using 1d timeframe for low trade frequency, Donchian for structure,
    # 1w ATR for volatility regime (avoid low vol whipsaws), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # first element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Align weekly ATR to 1d
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Get daily volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Volatility regime filter: 1w ATR > 1.3x 50-period average
    atr_ma = np.full(len(atr_1w_aligned), np.nan)
    for i in range(50, len(atr_1w_aligned)):
        atr_ma[i] = np.mean(atr_1w_aligned[i-50:i])
    high_vol_regime = atr_1w_aligned > (1.3 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(high_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic: Donchian breakout + high vol regime + volume confirmation
        long_entry = (close[i] > donchian_high[i-1]) and high_vol_regime[i] and volume_spike[i]
        short_entry = (close[i] < donchian_low[i-1]) and high_vol_regime[i] and volume_spike[i]
        
        # Exit logic: opposite Donchian breakout or volatility contraction
        long_exit = (close[i] < donchian_low[i-1]) or not high_vol_regime[i]
        short_exit = (close[i] > donchian_high[i-1]) or not high_vol_regime[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_atr_volume_v1"
timeframe = "1d"
leverage = 1.0