#!/usr/bin/env python3
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
    
    # === 1d Donchian Channels (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper channel: highest high of last 20 days
    upper_20 = np.full_like(high_1d, np.nan)
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
    
    # Lower channel: lowest low of last 20 days
    lower_20 = np.full_like(low_1d, np.nan)
    for i in range(19, len(low_1d)):
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Middle channel: average of upper and lower
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # === 1d ADX (14-period) for trend strength ===
    # Calculate +DM and -DM
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (14-period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_di_14 = 100 * wilders_smoothing(plus_dm, 14) / atr_14
    minus_di_14 = 100 * wilders_smoothing(minus_dm, 14) / atr_14
    dx = np.where((plus_di_14 + minus_di_14) > 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    adx_14 = wilders_smoothing(dx, 14)
    
    # Align all indicators to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # === 1d RSI (14-period) for mean reversion filter ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def wilders_smoothing_rsi(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing_rsi(gain, 14)
    avg_loss = wilders_smoothing_rsi(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        vol_ma_20 = np.zeros_like(volume)
        for j in range(len(volume)):
            if j >= 19:
                vol_ma_20[j] = np.mean(volume[j-19:j+1])
            else:
                vol_ma_20[j] = np.mean(volume[max(0, j-9):j+1]) if j > 0 else volume[0]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian + strong trend (ADX>25) + volume + RSI not overbought
            if (close[i] > upper_20_aligned[i] and 
                adx_14_aligned[i] > 25 and 
                vol_confirm and
                rsi_14_aligned[i] < 70):
                signals[i] = 0.30
                position = 1
                continue
            # Short: price breaks below lower Donchian + strong trend (ADX>25) + volume + RSI not oversold
            elif (close[i] < lower_20_aligned[i] and 
                  adx_14_aligned[i] > 25 and 
                  vol_confirm and
                  rsi_14_aligned[i] > 30):
                signals[i] = -0.30
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to middle Donchian OR trend weakens (ADX<20) OR RSI overbought
            if (close[i] < middle_20_aligned[i] or 
                adx_14_aligned[i] < 20 or
                rsi_14_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price returns to middle Donchian OR trend weakens (ADX<20) OR RSI oversold
            if (close[i] > middle_20_aligned[i] or 
                adx_14_aligned[i] < 20 or
                rsi_14_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.30
    
    return signals

name = "Donchian_ADX_RSI_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0