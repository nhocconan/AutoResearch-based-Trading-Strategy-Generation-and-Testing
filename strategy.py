#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume and 1w Trend Filter
Long when price breaks above Donchian upper band with above-average volume and 1w uptrend
Short when price breaks below Donchian lower band with above-average volume and 1w downtrend
Exit when price crosses Donchian median (midpoint) or volatility spike
Designed to work in both bull and bear markets by filtering with 1w trend and volume
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    # Calculate rolling max/min
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === 1-week Trend (HMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    hma_1w = calculate_hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Volatility filter (ATR 20) ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(hma_1w_aligned[i]) \
           or np.isnan(vol_ratio[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit: price crosses below median OR volatility spike (ATR > 2x average)
            if close[i] < donch_mid[i] or atr[i] > 2.0 * np.nanmean(atr[max(0, i-20):i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above median OR volatility spike
            if close[i] > donch_mid[i] or atr[i] > 2.0 * np.nanmean(atr[max(0, i-20):i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry conditions with 1w trend filter
            if close[i] > donch_high[i] and hma_1w_aligned[i] > hma_1w_aligned[i-1]:
                # Price breaks above upper band with 1w uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and hma_1w_aligned[i] < hma_1w_aligned[i-1]:
                # Price breaks below lower band with 1w downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculations
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        wma_vals = np.convolve(values, weights, mode='valid') / weights.sum()
        # Pad with NaN for beginning
        return np.concatenate([np.full(window-1, np.nan), wma_vals])
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Hull MA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final smoothing
    hma = wma(raw_hma, sqrt_period)
    
    return hma[-len(close):]  # Ensure same length as input