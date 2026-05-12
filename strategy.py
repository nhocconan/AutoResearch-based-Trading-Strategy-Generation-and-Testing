#!/usr/bin/env python3
# 6h_RSI_Divergence_1dTrend
# Hypothesis: On 6h timeframe, identify bullish/bearish RSI divergences with 1-day trend filter.
# Bullish divergence: Price makes lower low, RSI makes higher low (potential reversal up).
# Bearish divergence: Price makes higher high, RSI makes lower high (potential reversal down).
# Enter long on bullish divergence when 1-day EMA50 is rising, short on bearish divergence when falling.
# Exit when RSI crosses above 70 (long) or below 30 (short) or opposite divergence occurs.
# Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
# Targets 50-150 total trades over 4 years with disciplined entries.

name = "6h_RSI_Divergence_1dTrend"
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
    
    # Calculate RSI (14) on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_slope = np.diff(daily_ema50, prepend=0)  # positive = rising, negative = falling
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    daily_ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50_slope)
    
    # Identify pivot points for divergence detection
    # Look for local minima/maxima in price and RSI over 5-bar window
    window = 5
    def find_pivots(arr, window):
        n = len(arr)
        high_idx = np.argmax(np.lib.stride_tricks.sliding_window_view(arr, 2*window+1), axis=1) + np.arange(n) - window
        low_idx = np.argmin(np.lib.stride_tricks.sliding_window_view(arr, 2*window+1), axis=1) + np.arange(n) - window
        # Keep only valid indices within bounds
        high_idx = np.where((high_idx >= window) & (high_idx < n - window), high_idx, -1)
        low_idx = np.where((low_idx >= window) & (low_idx < n - window), low_idx, -1)
        return high_idx, low_idx
    
    # Simpler approach: track recent highs/lows
    lookback = 10
    price_low = np.full(n, np.nan)
    price_high = np.full(n, np.nan)
    rsi_low = np.full(n, np.nan)
    rsi_high = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Price low in lookback window
        price_low[i] = np.min(low[i-lookback:i+1])
        # Price high in lookback window
        price_high[i] = np.max(high[i-lookback:i+1])
        # RSI low in lookback window
        rsi_low[i] = np.min(rsi[i-lookback:i+1])
        # RSI high in lookback window
        rsi_high[i] = np.max(rsi[i-lookback:i+1])
    
    # Detect divergences
    bullish_div = np.zeros(n, dtype=bool)  # Price LL, RSI HL
    bearish_div = np.zeros(n, dtype=bool)  # Price HH, RSI LH
    
    for i in range(lookback*2, n):
        # Bullish divergence: current price low == lookback low AND current RSI > lookback RSI low
        if low[i] == price_low[i] and rsi[i] > rsi_low[i]:
            bullish_div[i] = True
        # Bearish divergence: current price high == lookback high AND current RSI < lookback RSI high
        if high[i] == price_high[i] and rsi[i] < rsi_high[i]:
            bearish_div[i] = True
    
    # Align divergence signals
    bullish_div_aligned = align_htf_to_ltf(prices, pd.DataFrame({'temp': range(n)}), bullish_div.astype(float))
    bearish_div_aligned = align_htf_to_ltf(prices, pd.DataFrame({'temp': range(n)}), bearish_div.astype(float))
    
    # Volume confirmation: current volume > 1.2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback*2, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(daily_ema50_aligned[i]) or np.isnan(daily_ema50_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        bull_div = bullish_div_aligned[i] > 0.5
        bear_div = bearish_div_aligned[i] > 0.5
        daily_trend_up = daily_ema50_slope_aligned[i] > 0
        daily_trend_down = daily_ema50_slope_aligned[i] < 0
        vol_confirm = volume_confirm[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # LONG: Bullish divergence with uptrend and volume
            if bull_div and daily_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence with downtrend and volume
            elif bear_div and daily_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or bearish divergence
            if rsi_val >= 70 or bear_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or bullish divergence
            if rsi_val <= 30 or bull_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals