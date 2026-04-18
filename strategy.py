#!/usr/bin/env python3
"""
4h_RSI_Divergence_Pattern_Recognition_v1
Hypothesis: Detect bullish and bearish RSI divergences on 4h timeframe using price extremes and RSI extremes. 
Bullish: price makes lower low while RSI makes higher low. Bearish: price makes higher high while RSI makes lower high.
Add volume confirmation (volume > 1.5x 20-period average) and 1-day trend filter (price > 50 EMA for longs, < 50 EMA for shorts).
This pattern works in both bull and bear markets as it identifies momentum exhaustion and potential reversals.
Target: 20-40 trades/year to avoid fee drag.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI (14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15]) if not np.isnan(gain).all() else 0
    avg_loss[14] = np.mean(loss[1:15]) if not np.isnan(loss).all() else 0
    
    for i in range(15, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily 50 EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[0] = close_1d[0]
    alpha_ema = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_50_1d[i] = alpha_ema * close_1d[i] + (1 - alpha_ema) * ema_50_1d[i-1]
    
    # Align daily EMA to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Precompute volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Check for RSI divergence patterns (look back 5 bars for swing points)
        lookback = 5
        if i >= lookback:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-lookback] and 
                rsi[i] > rsi[i-lookback] and
                vol_confirm and
                close[i] > ema_50_1d_aligned[i]):  # only long in uptrend (price > EMA50)
                if position <= 0:  # only enter if flat or short
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25  # maintain long
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (high[i] > high[i-lookback] and 
                  rsi[i] < rsi[i-lookback] and
                  vol_confirm and
                  close[i] < ema_50_1d_aligned[i]):  # only short in downtrend (price < EMA50)
                if position >= 0:  # only enter if flat or long
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25  # maintain short
        
        # Exit conditions: divergence fails or opposite signal
        elif position == 1:
            # Exit long if bearish divergence appears or price falls below EMA50
            if (high[i] > high[i-lookback] and 
                rsi[i] < rsi[i-lookback] and
                vol_confirm and
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25  # maintain long
        
        elif position == -1:
            # Exit short if bullish divergence appears or price rises above EMA50
            if (low[i] < low[i-lookback] and 
                rsi[i] > rsi[i-lookback] and
                vol_confirm and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25  # maintain short
    
    return signals

name = "4h_RSI_Divergence_Pattern_Recognition_v1"
timeframe = "4h"
leverage = 1.0