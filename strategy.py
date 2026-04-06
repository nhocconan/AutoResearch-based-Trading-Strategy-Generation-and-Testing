#!/usr/bin/env python3
"""
6H RSI DIVERGENCE + 12H TREND FILTER + VOLUME CONFIRMATION
Hypothesis: RSI divergences on 6h signal reversals. 12h trend filter ensures trades align with higher timeframe momentum. Volume confirms divergence strength. Works in bull (buy bullish divergences in uptrend) and bear (sell bearish divergences in downtrend). Target: 80-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_divergence_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def find_divergences(price, rsi_vals, lookback=10):
    """Find bullish and bearish RSI divergences"""
    n = len(price)
    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Look for bullish divergence: lower low in price, higher low in RSI
        if i >= lookback:
            # Find recent price low
            price_low_idx = np.argmin(price[i-lookback:i]) + i - lookback
            # Find RSI at that point
            rsi_at_price_low = rsi_vals[price_low_idx]
            # Current RSI
            rsi_now = rsi_vals[i]
            
            # Check if current price is lower than price at price_low_idx
            # and current RSI is higher than RSI at price_low_idx
            if (price[i] < price[price_low_idx] and 
                rsi_now > rsi_at_price_low):
                # Additional check: RSI should be in oversold territory (< 40)
                if rsi_now < 40:
                    bull_div[i] = True
        
        # Look for bearish divergence: higher high in price, lower high in RSI
        if i >= lookback:
            # Find recent price high
            price_high_idx = np.argmax(price[i-lookback:i]) + i - lookback
            # Find RSI at that point
            rsi_at_price_high = rsi_vals[price_high_idx]
            # Current RSI
            rsi_now = rsi_vals[i]
            
            # Check if current price is higher than price at price_high_idx
            # and current RSI is lower than RSI at price_high_idx
            if (price[i] > price[price_high_idx] and 
                rsi_now < rsi_at_price_high):
                # Additional check: RSI should be in overbought territory (> 60)
                if rsi_now > 60:
                    bear_div[i] = True
    
    return bull_div, bear_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 12h data for trend filter and RSI
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA(50) on 12h for trend filter
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_50_12h = ema(close_12h, 50)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate RSI(14) on 12h
    rsi_12h = rsi(close_12h, 14)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate RSI(14) on 6h for divergence detection
    rsi_6h = rsi(close, 14)
    
    # Find divergences on 6s
    bull_div_6h, bear_div_6h = find_divergences(close, rsi_6h, lookback=10)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bearish divergence or stoploss hit
            if (bear_div_6h[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bullish divergence or stoploss hit
            if (bull_div_6h[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: bullish divergence, above 12h EMA50, RSI < 50 (not overbought), with volume
            if (bull_div_6h[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                rsi_12h_aligned[i] < 50 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish divergence, below 12h EMA50, RSI > 50 (not oversold), with volume
            elif (bear_div_6h[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  rsi_12h_aligned[i] > 50 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals