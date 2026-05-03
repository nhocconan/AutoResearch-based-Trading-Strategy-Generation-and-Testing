#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high in 1d uptrend (ADX>25) with volume spike (>2.0x 20-period volume MA).
# Short when price breaks below 20-period Donchian low in 1d downtrend (ADX>25) with volume spike.
# Uses 1d ADX for higher timeframe trend strength to avoid whipsaw in ranging markets.
# Volume spike confirms institutional participation. Designed for 6h timeframe to achieve 50-150 total trades over 4 years.

name = "6h_Donchian20_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian(20) high and low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe (1d -> 6h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d ADX(14) for trend strength filter
    # ADX calculation: +DM, -DM, TR, then smoothed
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(np.abs(high_1d[1:] - high_1d[:-1]), np.abs(low_1d[1:] - low_1d[:-1]))
    )
    # Prepend first values to maintain array length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    plus_di_14 = 100 * wilders_smoothing(plus_dm, period_adx) / wilders_smoothing(tr, period_adx)
    minus_di_14 = 100 * wilders_smoothing(minus_dm, period_adx) / wilders_smoothing(tr, period_adx)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align ADX to lower timeframe (1d -> 6h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_strong = adx_aligned[i] > 25  # Strong trend filter
        
        if position == 0:
            # Long: price breaks above Donchian high AND strong trend AND volume spike
            if close_val > donchian_high_aligned[i] and trend_strong and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below Donchian low AND strong trend AND volume spike
            elif close_val < donchian_low_aligned[i] and trend_strong and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price breaks below Donchian low (opposite level)
            if close_val < donchian_low_aligned[i]:
                exit_signal = True
            # Exit: trend weakens (ADX < 20)
            elif adx_aligned[i] < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price breaks above Donchian high (opposite level)
            if close_val > donchian_high_aligned[i]:
                exit_signal = True
            # Exit: trend weakens (ADX < 20)
            elif adx_aligned[i] < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals