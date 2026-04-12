#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
    # Donchian breakout captures strong momentum; EMA50 filter avoids counter-trend trades
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Designed for low trade frequency (target: 20-40/year) to minimize fee drag
    # Works in bull/bear markets by only trading with higher-timeframe trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian Channel: 20-period high/low
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h volume for confirmation
    vol_ma_4h = np.full(len(df_4h), np.nan)
    for i in range(20, len(df_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i-20:i])
    
    # Volume confirmation: volume > 2.0 * 20-period average (4h)
    volume_spike_4h = volume_4h > (2.0 * vol_ma_4h)
    
    # Align all indicators to LTF (4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > donchian_high_aligned[i]
        bearish_breakout = close[i] < donchian_low_aligned[i]
        
        # 1d trend filter
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Donchian breakout + trend filter + volume confirmation
        long_entry = False
        short_entry = False
        
        # Long: bullish Donchian breakout + bullish 1d trend + volume spike
        if bullish_breakout and bullish_trend:
            long_entry = volume_spike_aligned[i]
        # Short: bearish Donchian breakout + bearish 1d trend + volume spike
        elif bearish_breakout and bearish_trend:
            short_entry = volume_spike_aligned[i]
        
        # Exit logic: opposite Donchian breakout or trend reversal
        long_exit = bearish_breakout or not bullish_trend
        short_exit = bullish_breakout or not bearish_trend
        
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

name = "4h_1d_donchian_trend_volume_v1"
timeframe = "4h"
leverage = 1.0