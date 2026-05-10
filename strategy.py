#!/usr/bin/env python3
# 6h_Triple_RSI_Divergence_1dTrend_Volume
# Hypothesis: Combines RSI divergence detection with higher-timeframe trend and volume confirmation.
# Uses 3-period RSI for sensitivity to short-term exhaustion, confirmed by daily trend filter.
# Works in bull markets by buying oversold dips in uptrends, and in bear markets by selling
# overbought rallies in downtrends. Volume confirmation reduces false signals. Designed for
# low frequency (~15-30 trades/year) to minimize fee drag on 6h timeframe.

name = "6h_Triple_RSI_Divergence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with given period."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_divergence(price, rsi, lookback=10):
    """Detect bullish and bearish divergence."""
    bullish_div = np.zeros_like(price, dtype=bool)
    bearish_div = np.zeros_like(price, dtype=bool)
    
    for i in range(lookback, len(price)):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (price[i] < price[i-lookback] and 
            rsi[i] > rsi[i-lookback]):
            # Check if this is a meaningful low point
            if np.all(price[i-lookback:i] >= price[i]) and np.all(rsi[i-lookback:i] <= rsi[i]):
                bullish_div[i] = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (price[i] > price[i-lookback] and 
            rsi[i] < rsi[i-lookback]):
            # Check if this is a meaningful high point
            if np.all(price[i-lookback:i] <= price[i]) and np.all(rsi[i-lookback:i] >= rsi[i]):
                bearish_div[i] = True
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 3-period RSI for sensitivity
    rsi_3 = calculate_rsi(close, 3)
    
    # Daily trend: EMA34 on daily close
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Detect RSI divergence
    bullish_div, bearish_div = detect_divergence(close, rsi_3, lookback=8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_3[i]) or 
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: bullish RSI divergence with daily uptrend and volume
            if (bullish_div[i] and 
                trend_1d_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish RSI divergence with daily downtrend and volume
            elif (bearish_div[i] and 
                  trend_1d_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when bearish divergence appears or trend fails
            if (bearish_div[i] or 
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when bullish divergence appears or trend fails
            if (bullish_div[i] or 
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals