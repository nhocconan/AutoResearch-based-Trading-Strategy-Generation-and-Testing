#!/usr/bin/env python3
# 4h_RSI_Divergence_Divergence_1dTrend_Volume
# Hypothesis: On 4h timeframe, RSI divergence with daily trend filter and volume confirmation captures reversals in both bull and bear markets. Divergence identifies exhaustion, daily trend avoids counter-trend trades, volume reduces false signals. Designed for low frequency (~20-50 trades/year) to minimize fee drag.

name = "4h_RSI_Divergence_Divergence_1dTrend_Volume"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # RSI (14)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily trend: EMA34 on daily close
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # RSI divergence detection (lookback 5 periods)
    def find_divergence(high_vals, low_vals, rsi_vals, lookback=5):
        bullish_div = np.zeros(len(close), dtype=bool)
        bearish_div = np.zeros(len(close), dtype=bool)
        
        for i in range(lookback, len(close)):
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low_vals[i] < low_vals[i-lookback] and 
                rsi_vals[i] > rsi_vals[i-lookback]):
                # Check if it's a meaningful low
                if low_vals[i] == np.min(low_vals[i-lookback:i+1]):
                    bullish_div[i] = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (high_vals[i] > high_vals[i-lookback] and 
                rsi_vals[i] < rsi_vals[i-lookback]):
                # Check if it's a meaningful high
                if high_vals[i] == np.max(high_vals[i-lookback:i+1]):
                    bearish_div[i] = True
        
        return bullish_div, bearish_div
    
    bullish_div, bearish_div = find_divergence(high, low, rsi_values, lookback=5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_values[i]) or 
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
            # Exit when RSI returns to neutral (50) or trend fails
            if (rsi_values[i] >= 50 or 
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when RSI returns to neutral (50) or trend fails
            if (rsi_values[i] <= 50 or 
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals