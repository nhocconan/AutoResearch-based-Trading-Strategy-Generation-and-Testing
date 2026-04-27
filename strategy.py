#!/usr/bin/env python3
"""
6h Williams Fractal + EMA Trend + Volume Strategy
Hypothesis: Williams Fractal identifies turning points, EMA (21) confirms trend direction,
and volume spikes filter false breakouts. Works in both bull/bear markets by only
taking trades in direction of higher timeframe trend.
Target: 50-150 trades over 4 years (~12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (up) and bullish (down)"""
    n = len(high)
    bearish = np.full(n, np.nan)  # Up fractal (peak)
    bullish = np.full(n, np.nan)  # Down fractal (valley)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        # Bullish fractal: low[i] is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractal and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    
    # Calculate EMA(21) on daily close
    close_1d = df_1d['close'].values
    ema_21 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 21:
        ema_21[20:] = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values[20:]
    
    # Williams Fractals need 2 extra bars for confirmation (Williams uses 2-bar confirmation)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Calculate 20-period ATR for stop loss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if np.all(~np.isnan(tr[i-13:i+1])):
            atr[i] = np.nanmean(tr[i-13:i+1])
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period: need enough data for all indicators
    start_idx = max(20, 19, 2)  # EMA(21), volume MA(20), fractal lookback
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_confirmed[i]) or
            np.isnan(bullish_fractal_confirmed[i]) or
            np.isnan(ema_21_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below daily EMA(21)
        bullish_trend = price > ema_21_aligned[i]
        bearish_trend = price < ema_21_aligned[i]
        
        # Volume confirmation: spike > 1.8x average
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long entry: bullish fractal (support) + bullish trend + volume
            if (not np.isnan(bullish_fractal_confirmed[i]) and 
                bullish_trend and volume_confirmation):
                signals[i] = size
                position = 1
            # Short entry: bearish fractal (resistance) + bearish trend + volume
            elif (not np.isnan(bearish_fractal_confirmed[i]) and 
                  bearish_trend and volume_confirmation):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bearish fractal (resistance) or trend turns bearish
            if (not np.isnan(bearish_fractal_confirmed[i]) or not bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: bullish fractal (support) or trend turns bullish
            if (not np.isnan(bullish_fractal_confirmed[i]) or not bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_EMATrend_Volume"
timeframe = "6h"
leverage = 1.0