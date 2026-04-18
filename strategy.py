#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Pullback_With_Volume
Hypothesis: On 12h timeframe, use Kaufman's Adaptive Moving Average (KAMA) for trend direction,
RSI for pullback entries, and volume confirmation to filter false signals.
KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI pullbacks
provide entries in the direction of the trend during temporary countertrend moves.
Volume confirmation ensures institutional participation. Designed for low trade
frequency (12-37/year) to avoid fee drag while capturing trending moves in both
bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix: calculate volatility correctly using rolling sum
    volatility = pd.Series(np.abs(np.diff(close))).rolling(window=er_period, min_periods=1).sum().values
    volatility = np.concatenate([np.full(er_period-1, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, len(close)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 12h timeframe (already on 12h, no alignment needed)
    # But we need 1d trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # RSI(14) for pullback entries
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = np.nan
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(200, 14, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(ema_200_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        ema200 = ema_200_aligned[i]
        rsi_val = rsi[i]
        vol_conf = volume_confirmed[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI pulled back to oversold, volume confirmation
            if price > kama_val and rsi_val < 30 and vol_conf and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI pulled back to overbought, volume confirmation
            elif price < kama_val and rsi_val > 70 and vol_conf and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR RSI overbought
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR RSI oversold
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Direction_RSI_Pullback_With_Volume"
timeframe = "12h"
leverage = 1.0