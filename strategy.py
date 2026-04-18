#!/usr/bin/env python3
"""
4h_RSI_Divergence_BullBear_Mode
Hypothesis: 4-hour RSI divergences (bullish/bearish) with volume confirmation and ADX trend filter.
Exploits exhaustion points in trends - works in bull markets (buy dips) and bear markets (sell rallies).
Uses RSI(14) for momentum, ADX(14) for trend strength (>25), and volume spike (>1.5x avg) for confirmation.
Target: 20-40 trades/year to minimize fee drag while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper Wilder's smoothing."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(prices, np.nan)
    avg_loss = np.full_like(prices, np.nan)
    
    # Initial average
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed averages
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    
    if len(dx) >= period:
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def find_rsi_divergence(prices, rsi, lookback=14):
    """Find bullish and bearish RSI divergences."""
    n = len(prices)
    bullish_div = np.full(n, False)
    bearish_div = np.full(n, False)
    
    for i in range(lookback, n):
        # Look for price making lower low while RSI makes higher low (bullish div)
        if i >= lookback:
            price_low_idx = np.argmin(prices[i-lookback:i+1]) + i - lookback
            rsi_low_idx = np.argmin(rsi[i-lookback:i+1]) + i - lookback
            
            # Bullish divergence: price lower low, RSI higher low
            if (prices[price_low_idx] < prices[i-lookback] and 
                rsi[rsi_low_idx] > rsi[i-lookback] and
                price_low_idx != rsi_low_idx):
                bullish_div[i] = True
            
            # Bearish divergence: price higher high, RSI lower high
            price_high_idx = np.argmax(prices[i-lookback:i+1]) + i - lookback
            rsi_high_idx = np.argmax(rsi[i-lookback:i+1]) + i - lookback
            
            if (prices[price_high_idx] > prices[i-lookback] and 
                rsi[rsi_high_idx] < rsi[i-lookback] and
                price_high_idx != rsi_high_idx):
                bearish_div[i] = True
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    rsi = calculate_rsi(close, 14)
    
    # Calculate ADX(14) for trend filter
    adx = calculate_adx(high, low, close, 14)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Find RSI divergences
    bullish_div, bearish_div = find_rsi_divergence(close, rsi, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (trending market)
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: bullish RSI divergence + volume spike + strong trend
            if (bullish_div[i] and vol_spike[i] and strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence + volume spike + strong trend
            elif (bearish_div[i] and vol_spike[i] and strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or trend weakens
            if (rsi[i] > 70 or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or trend weakens
            if (rsi[i] < 30 or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Divergence_BullBear_Mode"
timeframe = "4h"
leverage = 1.0