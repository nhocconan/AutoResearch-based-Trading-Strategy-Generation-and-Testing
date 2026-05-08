#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX with 1d price action patterns and volume confirmation.
# Long when ADX > 25 (trending), price > 1d open (bullish daily bias), and volume > 1.5x 20-period average.
# Short when ADX > 25 (trending), price < 1d open (bearish daily bias), and volume > 1.5x 20-period average.
# Exit when ADX falls below 20 (trend weakening) to avoid whipsaw in ranging markets.
# Uses ADX for trend strength, daily open bias for direction, and volume for confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "6h_ADX_1dOpenBias_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for open bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 6h data
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    atr = np.zeros_like(close)
    atr[0] = tr[0] if len(tr) > 0 else 0
    for i in range(1, len(tr)):
        atr[i+1] = (atr[i] * 13 + tr[i]) / 14
    plus_di = 100 * np.where(atr[14:] != 0, np.cumsum(plus_dm)[13:] / atr[14:], 0)
    minus_di = 100 * np.where(atr[14:] != 0, np.cumsum(minus_dm)[13:] / atr[14:], 0)
    dx = 100 * np.where((plus_di + minus_di) != 0, np.absolute(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(close)
    adx[27:] = np.convolve(dx, np.ones(14)/14, mode='valid')[:len(adx)-27]
    # Pad beginning with zeros for simplicity (will be handled by min_periods logic)
    adx = np.concatenate([np.zeros(27), adx[:len(adx)-27]]) if len(adx) > 27 else np.zeros(len(close))
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d open bias: today's close vs today's open
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_bias = close_1d > open_1d  # True for bullish day, False for bearish
    daily_bias_aligned = align_htf_to_ltf(prices, df_1d, daily_bias.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(daily_bias_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ADX > 25 (strong trend), bullish daily bias, volume spike
            long_cond = (adx[i] > 25) and daily_bias_aligned[i] and volume_filter[i]
            # Short conditions: ADX > 25 (strong trend), bearish daily bias, volume spike
            short_cond = (adx[i] > 25) and (not daily_bias_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX falls below 20 (trend weakening)
            if adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX falls below 20 (trend weakening)
            if adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals