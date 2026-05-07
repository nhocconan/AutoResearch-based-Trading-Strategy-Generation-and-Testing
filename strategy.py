#!/usr/bin/env python3
# 4h_RSI_Divergence_Volume_1dTrend
# Hypothesis: RSI divergence (price makes new high/low while RSI does not) combined with volume confirmation and 1-day trend filter.
# In bull markets (price > 1d EMA50), look for bullish RSI divergence for long entries.
# In bear markets (price < 1d EMA50), look for bearish RSI divergence for short entries.
# Volume confirmation reduces false signals. Target: 20-30 trades per year (~80-120 over 4 years) with position size 0.25.

name = "4h_RSI_Divergence_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # RSI divergence detection
    # Bullish divergence: price makes new low, RSI makes higher low
    # Bearish divergence: price makes new high, RSI makes lower high
    lookback = 10  # look back 10 periods for swing points
    
    # Find recent swing lows and highs
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Check for bullish divergence: price low lower, RSI low higher
        if low[i] == np.min(low[i-lookback:i+1]) and rsi[i] > np.min(rsi[i-lookback:i+1]):
            # Find if there was a previous low in the lookback window
            prev_low_idx = np.argmin(low[i-lookback:i]) + (i-lookback)
            if low[i] < low[prev_low_idx] and rsi[i] > rsi[prev_low_idx]:
                bullish_div[i] = True
        
        # Check for bearish divergence: price high higher, RSI high lower
        if high[i] == np.max(high[i-lookback:i+1]) and rsi[i] < np.max(rsi[i-lookback:i+1]):
            # Find if there was a previous high in the lookback window
            prev_high_idx = np.argmax(high[i-lookback:i]) + (i-lookback)
            if high[i] > high[prev_high_idx] and rsi[i] < rsi[prev_high_idx]:
                bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA50
        uptrend_regime = close[i] > ema_50_1d_aligned[i]
        downtrend_regime = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: bullish RSI divergence in uptrend regime + volume
            long_entry = bullish_div[i] and uptrend_regime and volume_confirm
            # Short: bearish RSI divergence in downtrend regime + volume
            short_entry = bearish_div[i] and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below 1-day EMA50 or bearish divergence appears
            if (close[i] < ema_50_1d_aligned[i]) or bearish_div[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above 1-day EMA50 or bullish divergence appears
            if (close[i] > ema_50_1d_aligned[i]) or bullish_div[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals