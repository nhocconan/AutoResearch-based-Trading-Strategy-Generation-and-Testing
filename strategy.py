#!/usr/bin/env python3
"""
12h_1d_1w_Trend_With_Volume_Confirmation
Hypothesis: Combines 1d trend (HMA) with 12h price action and volume confirmation. 
Uses weekly regime filter to avoid counter-trend trades in strong trends.
Designed to work in both bull and bear markets by following the dominant trend.
Target: 12-37 trades/year on 12h (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).rolling(window=half_period, min_periods=half_period).mean()
    wma2 = pd.Series(series).rolling(window=period, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on daily for trend
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on weekly for regime
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w, additional_delay_bars=1)
    
    # 12h price action
    # Calculate 12h ATR for breakout threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(hma_21_1d_aligned[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or \
           np.isnan(atr[i]) or \
           np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Trend condition: price vs HMA
        bullish_trend = close[i] > hma_21_1d_aligned[i]
        bearish_trend = close[i] < hma_21_1d_aligned[i]
        
        # Regime filter: price vs weekly EMA
        bullish_regime = close[i] > ema_50_1w_aligned[i]
        bearish_regime = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > vol_ma[i]
        
        # Breakout magnitude
        breakout_threshold = 0.5 * atr[i]
        
        # Entry logic
        if bullish_trend and bullish_regime and volume_ok:
            if close[i] > close[i-1] + breakout_threshold:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
        elif bearish_trend and bearish_regime and volume_ok:
            if close[i] < close[i-1] - breakout_threshold:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
        else:
            # Exit conditions
            if position == 1 and (not bullish_trend or not bullish_regime or not volume_ok):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not bearish_trend or not bearish_regime or not volume_ok):
                position = 0
                signals[i] = 0.0
            elif position == 0:
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = position_size if position == 1 else -position_size
    
    return signals

name = "12h_1d_1w_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0