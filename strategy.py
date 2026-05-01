#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND 1d ADX > 25 AND volume > 1.5x 12h volume median.
# Short when price breaks below Donchian(20) lower band AND 1d ADX > 25 AND volume > 1.5x 12h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Donchian provides objective structure, 1d ADX filters for trending markets (works in both bull/bear),
# volume confirmation ensures momentum breakout validity. Target: 12-25 trades/year on 12h timeframe.
# Proven pattern: Donchian breakouts with trend/volume filters work on BTC/ETH in all regimes.

name = "12h_Donchian20_1dADX_Volume_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume median (20-period for stability)
    vol_median_12h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 1d ADX(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # True Range for ADX
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
            np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
        )
    )
    # Directional Movement
    dm_plus = np.where(
        (df_1d['high'].values - np.concatenate([[df_1d['high'].values[0]], df_1d['high'].values[:-1]])) >
        (np.concatenate([[df_1d['low'].values[0]], df_1d['low'].values[:-1]]) - df_1d['low'].values),
        np.maximum(df_1d['high'].values - np.concatenate([[df_1d['high'].values[0]], df_1d['high'].values[:-1]]), 0),
        0
    )
    dm_minus = np.where(
        (np.concatenate([[df_1d['low'].values[0]], df_1d['low'].values[:-1]]) - df_1d['low'].values) >
        (df_1d['high'].values - np.concatenate([[df_1d['high'].values[0]], df_1d['high'].values[:-1]])),
        np.maximum(np.concatenate([[df_1d['low'].values[0]], df_1d['low'].values[:-1]]) - df_1d['low'].values, 0),
        0
    )
    # Smooth TR, DM+, DM- with Welles Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = WilderSmoothing(tr_1d, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = WilderSmoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian(20) channels from 1d data
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Upper band: 20-period high
    upper_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lower_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, ADX, volume, and Donchian
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(vol_median_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 12h volume median
        if vol_median_12h[i] <= 0 or np.isnan(vol_median_12h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_12h[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > upper band AND trending AND volume spike
            if curr_close > upper_aligned[i] and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < lower band AND trending AND volume spike
            elif curr_close < lower_aligned[i] and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below lower band OR ADX weakens (< 20)
            elif curr_close < lower_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above upper band OR ADX weakens (< 20)
            elif curr_close > upper_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals