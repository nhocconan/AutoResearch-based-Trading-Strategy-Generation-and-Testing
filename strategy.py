#!/usr/bin/env python3
# 6h_RSI_Divergence_TopBottom_1dTrend
# Hypothesis: Detect bullish/bearish RSI divergences on 6s with 1d trend filter.
# Bullish: price makes lower low while RSI makes higher low (oversold reversal).
# Bearish: price makes higher high while RSI makes lower high (overbought reversal).
# Enter only in direction of 1d EMA50 trend to avoid counter-trend trades in strong trends.
# Volume confirmation ensures divergence has participation.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "6h_RSI_Divergence_TopBottom_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === RSI(14) on 6h ===
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # RSI needs warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Initialize divergence flags
        bullish_div = False
        bearish_div = False
        
        # Look for bullish divergence: price LL, RSI HL
        if i >= 20:  # Need lookback
            # Find recent low in price and RSI
            price_low_idx = i - np.argmin(low[i-19:i+1])  # lookback 20 bars
            rsi_low_idx = i - np.argmin(rsi[i-19:i+1])
            
            # Bullish div: price makes lower low, RSI makes higher low
            if (price_low_idx != rsi_low_idx and 
                low[i] < low[price_low_idx] and 
                rsi[i] > rsi[rsi_low_idx] and
                rsi[i] < 40):  # Oversold condition
                bullish_div = True
        
        # Look for bearish divergence: price HH, RSI LH
        if i >= 20:
            # Find recent high in price and RSI
            price_high_idx = i - np.argmax(high[i-19:i+1])
            rsi_high_idx = i - np.argmax(rsi[i-19:i+1])
            
            # Bearish div: price makes higher high, RSI makes lower high
            if (price_high_idx != rsi_high_idx and 
                high[i] > high[price_high_idx] and 
                rsi[i] < rsi[rsi_high_idx] and
                rsi[i] > 60):  # Overbought condition
                bearish_div = True
        
        # Entry logic
        if position == 0:
            # LONG: bullish divergence + uptrend + volume
            if bullish_div and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish divergence + downtrend + volume
            elif bearish_div and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: bearish divergence or trend breaks
            if bearish_div or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish divergence or trend breaks
            if bullish_div or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals