#!/usr/bin/env python3
"""
Hypothesis: 12h RSI divergence with 1d volume-weighted price action and ADX trend filter.
- Long: RSI(14) bullish divergence (price makes lower low, RSI makes higher low) + price > VWAP(20) + ADX > 20
- Short: RSI(14) bearish divergence (price makes higher high, RSI makes lower high) + price < VWAP(20) + ADX > 20
- Exit: RSI crosses above 70 (long) or below 30 (short) or opposite divergence occurs
- Uses 1d VWAP and RSI for confluence, designed to work in both trending and ranging markets.
Target: 15-35 trades/year (60-140 total) to minimize fee drag while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    if len(gain) >= period:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if not np.isnan(rs[i]):
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index."""
    if len(high) < period * 2:
        return np.full(len(high), np.nan)
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    if len(dm_plus) >= period:
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        for i in range(period, len(dm_plus)):
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Calculate Directional Indicators
    plus_di = np.full(len(dm_plus), np.nan)
    minus_di = np.full(len(dm_minus), np.nan)
    for i in range(period, len(atr)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.full(len(plus_di), np.nan)
    for i in range(period, len(plus_di)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.full(len(dx), np.nan)
    if len(dx) >= 2 * period - 1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_vwap(high, low, close, volume, period):
    """Calculate Volume Weighted Average Price."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    typical_price = (high + low + close) / 3.0
    vwap = np.full(len(close), np.nan)
    
    for i in range(period-1, len(close)):
        start_idx = i - period + 1
        tp_sum = np.sum(typical_price[start_idx:i+1] * volume[start_idx:i+1])
        vol_sum = np.sum(volume[start_idx:i+1])
        if vol_sum != 0:
            vwap[i] = tp_sum / vol_sum
    
    return vwap

def find_divergences(price, rsi, lookback=5):
    """Find bullish and bearish divergences."""
    n = len(price)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        price_low = np.argmin(price[i-lookback:i+1]) + i - lookback
        rsi_low = np.argmin(rsi[i-lookback:i+1]) + i - lookback
        if price_low != rsi_low and price[price_low] < price[i-lookback] and rsi[rsi_low] > rsi[i-lookback]:
            # Check if current point is a low
            if i == np.argmin(price[i-lookback:i+1]) + i - lookback:
                bullish_div[i] = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        price_high = np.argmax(price[i-lookback:i+1]) + i - lookback
        rsi_high = np.argmax(rsi[i-lookback:i+1]) + i - lookback
        if price_high != rsi_high and price[price_high] > price[i-lookback] and rsi[rsi_high] < rsi[i-lookback]:
            # Check if current point is a high
            if i == np.argmax(price[i-lookback:i+1]) + i - lookback:
                bearish_div[i] = True
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP and RSI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate VWAP (20-period) on 1d
    vwap_20_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d, 20)
    
    # Calculate RSI (14-period) on 1d
    rsi_14_1d = calculate_rsi(close_1d, 14)
    
    # Calculate ADX (14-period) on 1d
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Find divergences on 1d RSI
    bullish_div_1d, bearish_div_1d = find_divergences(close_1d, rsi_14_1d, 5)
    
    # Align to 12h timeframe
    vwap_20_1d_12h = align_htf_to_ltf(prices, df_1d, vwap_20_1d)
    rsi_14_1d_12h = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    adx_14_1d_12h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    bullish_div_1d_12h = align_htf_to_ltf(prices, df_1d, bullish_div_1d.astype(float))
    bearish_div_1d_12h = align_htf_to_ltf(prices, df_1d, bearish_div_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need VWAP, RSI, ADX, and divergence data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_20_1d_12h[i]) or np.isnan(rsi_14_1d_12h[i]) or 
            np.isnan(adx_14_1d_12h[i]) or np.isnan(bullish_div_1d_12h[i]) or 
            np.isnan(bearish_div_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish divergence + price > VWAP + ADX > 20
            if bullish_div_1d_12h[i] and close[i] > vwap_20_1d_12h[i] and adx_14_1d_12h[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + price < VWAP + ADX > 20
            elif bearish_div_1d_12h[i] and close[i] < vwap_20_1d_12h[i] and adx_14_1d_12h[i] > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 70 or bearish divergence
            if rsi_14_1d_12h[i] > 70 or bearish_div_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 or bullish divergence
            if rsi_14_1d_12h[i] < 30 or bullish_div_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSIDivergence_VWAP_ADX"
timeframe = "12h"
leverage = 1.0