#!/usr/bin/env python3
"""
1d_1w_kama_volume_v1
Strategy: 1d KAMA trend with volume confirmation and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses KAMA on daily timeframe to capture trend direction, confirmed by volume spikes (>2x average volume), and filtered by weekly EMA21 trend. KAMA adapts to market noise, reducing false signals in choppy markets. Volume confirmation ensures institutional participation. Weekly trend filter avoids counter-trend trades. Designed to work in both bull and bear markets by following the dominant trend on higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_volume_v1"
timeframe = "1d"
leverage = 1.0

def kama(close, er_length=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 1 else np.convolve(np.abs(np.diff(close)), np.ones(er_length), 'same')
    # For 1D array
    volatility = np.array([np.sum(np.abs(np.diff(close[max(0, i-er_length+1):i+1]))) for i in range(len(close))])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # KAMA on daily timeframe
    kama_val = kama(close, er_length=10, fast=2, slow=30)
    
    # Weekly EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_val[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters: price above/below KAMA and weekly EMA
        above_kama = price_close > kama_val[i]
        above_weekly_ema = price_close > ema_21_1w_aligned[i]
        below_kama = price_close < kama_val[i]
        below_weekly_ema = price_close < ema_21_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: price above KAMA and weekly EMA with volume
        long_signal = above_kama and above_weekly_ema and vol_confirmed
        
        # Short: price below KAMA and weekly EMA with volume
        short_signal = below_kama and below_weekly_ema and vol_confirmed
        
        # Exit when price crosses KAMA in opposite direction
        exit_long = position == 1 and price_close < kama_val[i]
        exit_short = position == -1 and price_close > kama_val[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals