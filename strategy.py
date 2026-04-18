# 4h KAMA + RSI + Chop Filter
# Trend-following with adaptive mean (KAMA) and RSI momentum, filtered by choppy market conditions.
# Uses 1d ADX for trend strength confirmation to avoid false signals in ranging markets.
# Designed for 20-50 trades/year to minimize fee drag while capturing major trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average."""
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.copy(close)
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_chop(high, low, close, period=14):
    """Choppiness Index."""
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(tr)
    for i in range(period, len(tr)):
        atr[i] = np.sum(tr[i-period+1:i+1]) / period
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    for i in range(period-1, len(high)):
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    chop = np.zeros_like(close)
    for i in range(period-1, len(close)):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
        else:
            chop[i] = 50
    return chop

def calculate_adx(high, low, close, period):
    """Average Directional Index."""
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    dx = np.zeros_like(plus_di)
    dx_sum = plus_di + minus_di
    dx = np.where(dx_sum != 0, 100 * np.abs(plus_di - minus_di) / dx_sum, 0)
    adx = np.zeros_like(dx)
    for i in range(2*period-1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_chop(high, low, close, period=14)
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure all indicators are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(adx_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Market regime: chop < 61.8 = trending, chop > 61.8 = ranging
        trending_market = chop[i] < 61.8
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, trending market, ADX > 20
            if (close[i] > kama[i] and rsi[i] > 50 and 
                trending_market and adx_14_1d_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, trending market, ADX > 20
            elif (close[i] < kama[i] and rsi[i] < 50 and 
                  trending_market and adx_14_1d_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA or RSI < 40 or chop > 61.8 (ranging)
            if (close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA or RSI > 60 or chop > 61.8 (ranging)
            if (close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0