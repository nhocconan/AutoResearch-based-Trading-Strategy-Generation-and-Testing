#!/usr/bin/env python3
"""
4h_KAMA_Direction_Supertrend_Momentum
Hypothesis: Use KAMA to determine long-term trend direction, Supertrend for entry signals, and RSI for momentum filtering. Only take trades in direction of KAMA trend when Supertrend confirms and RSI shows sufficient momentum. This avoids counter-trend trades and whipsaws in choppy markets. Position size 0.25 targeting ~30 trades/year to minimize fee drag. Works in bull/bear by following trend with momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA trend filter (ER=10, fast=2, slow=30)
    def calculate_kama(close_prices, er_length=10, fast=2, slow=30):
        n = len(close_prices)
        kama = np.full(n, np.nan)
        if n < er_length + 1:
            return kama
        
        # Calculate efficiency ratio
        change = np.abs(np.diff(close_prices, er_length))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=1)
        er = np.zeros(n)
        er[er_length:] = change / np.maximum(volatility[er_length-1:], 1e-10)
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Initialize KAMA
        kama[er_length] = close_prices[er_length]
        for i in range(er_length + 1, n):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # Supertrend (ATR=10, multiplier=3.0)
    def calculate_supertrend(high_prices, low_prices, close_prices, atr_length=10, multiplier=3.0):
        n = len(close_prices)
        if n < atr_length:
            return np.full(n, np.nan), np.full(n, np.nan)
        
        # True Range
        tr1 = high_prices[1:] - low_prices[1:]
        tr2 = np.abs(high_prices[1:] - close_prices[:-1])
        tr3 = np.abs(low_prices[1:] - close_prices[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[high_prices[0] - low_prices[0]], tr])
        
        # ATR
        atr = np.full(n, np.nan)
        atr[atr_length-1] = np.mean(tr[:atr_length])
        for i in range(atr_length, n):
            atr[i] = (atr[i-1] * (atr_length-1) + tr[i]) / atr_length
        
        # Supertrend
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        supertrend = np.full(n, np.nan)
        
        for i in range(atr_length, n):
            upper_band[i] = (high_prices[i] + low_prices[i]) / 2 + multiplier * atr[i]
            lower_band[i] = (high_prices[i] + low_prices[i]) / 2 - multiplier * atr[i]
            
            if i == atr_length:
                supertrend[i] = upper_band[i]
            else:
                if supertrend[i-1] == upper_band[i-1]:
                    supertrend[i] = lower_band[i] if close[i] > upper_band[i-1] else upper_band[i]
                else:
                    supertrend[i] = upper_band[i] if close[i] < lower_band[i-1] else lower_band[i]
        
        return supertrend, atr
    
    # RSI momentum filter
    def calculate_rsi(close_prices, length=14):
        n = len(close_prices)
        if n < length + 1:
            return np.full(n, np.nan)
        
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        
        for i in range(length + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        rs = np.divide(avg_gain, avg_loss, out=np.full(n, np.nan), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    supertrend, atr = calculate_supertrend(high, low, close, atr_length=10, multiplier=3.0)
    rsi = calculate_rsi(close, length=14)
    
    # Volume confirmation
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(supertrend[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        rsi_min = 40 if close[i] > kama[i] else 60  # Higher RSI threshold for counter-trend
        
        if position == 0:
            # Long: price above KAMA, Supertrend long, RSI > 40, volume confirmation
            if close[i] > kama[i] and close[i] > supertrend[i] and rsi[i] > rsi_min and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, Supertrend short, RSI < 60, volume confirmation
            elif close[i] < kama[i] and close[i] < supertrend[i] and rsi[i] < (100 - rsi_min) and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Supertrend or below KAMA
            if close[i] < supertrend[i] or close[i] < kama[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Supertrend or above KAMA
            if close[i] > supertrend[i] or close[i] > kama[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_Supertrend_Momentum"
timeframe = "4h"
leverage = 1.0