#!/usr/bin/env python3
"""
12h_VWAP_MeanReversion_RangeBound
Hypothesis: On 12h timeframe, trade mean-reversion from VWAP in range-bound markets.
Long when price < VWAP - 0.5*ATR and short when price > VWAP + 0.5*ATR, filtered by
daily RSI < 40 (long) or > 60 (short) and low volatility (ATR < median ATR).
Exit when price crosses VWAP or reverses RSI signal.
Designed for low trade frequency (12-37/year) to work in both bull and bear markets
by capturing mean reversion in ranging conditions while avoiding trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    return smooth_wilder(tr, period)

def calculate_vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    return np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)

def calculate_rsi(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) < period:
        return avg_gain, avg_loss
        
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Data (HTF for VWAP, RSI, ATR) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR (14-period)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily median ATR (50-period) for volatility filter
    atr_median_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).median().values
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    
    # Daily VWAP
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Daily RSI (14-period)
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_median_1d_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volatility filter: only trade when ATR < median ATR (low volatility = ranging)
        low_vol = atr_1d_aligned[i] < atr_median_1d_aligned[i]
        
        # Mean reversion signals
        price_vwap_diff = close[i] - vwap_1d_aligned[i]
        long_signal = price_vwap_diff < -0.5 * atr_1d_aligned[i]
        short_signal = price_vwap_diff > 0.5 * atr_1d_aligned[i]
        
        # RSI filters: RSI < 40 for long, RSI > 60 for short
        rsi_long_filter = rsi_1d_aligned[i] < 40
        rsi_short_filter = rsi_1d_aligned[i] > 60
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price below VWAP - 0.5*ATR, RSI < 40, low volatility
            if long_signal and rsi_long_filter and low_vol:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price above VWAP + 0.5*ATR, RSI > 60, low volatility
            elif short_signal and rsi_short_filter and low_vol:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price crosses above VWAP or RSI > 50 (momentum shift)
            if close[i] > vwap_1d_aligned[i] or rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses below VWAP or RSI < 50 (momentum shift)
            if close[i] < vwap_1d_aligned[i] or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VWAP_MeanReversion_RangeBound"
timeframe = "12h"
leverage = 1.0