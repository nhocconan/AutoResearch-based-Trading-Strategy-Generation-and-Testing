#!/usr/bin/env python3
# 1d_weekly_rsi_divergence_v1
# Hypothesis: Weekly RSI divergence on price extremes with daily price action confirmation.
# In bull markets: buy when weekly RSI shows bullish divergence at support and price closes above daily VWAP.
# In bear markets: sell when weekly RSI shows bearish divergence at resistance and price closes below daily VWAP.
# Weekly timeframe reduces noise, RSI divergence captures exhaustion, VWAP filters for institutional participation.
# Target: 15-25 trades/year with ~0.25 position size to minimize fee drag.

name = "1d_weekly_rsi_divergence_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper Wilder's smoothing."""
    delta = np.diff(prices)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: first average is simple average
    avg_up = np.zeros_like(prices)
    avg_down = np.zeros_like(prices)
    avg_up[period] = np.mean(up[:period])
    avg_down[period] = np.mean(down[:period])
    
    for i in range(period + 1, len(prices)):
        avg_up[i] = (avg_up[i-1] * (period - 1) + up[i-1]) / period
        avg_down[i] = (avg_down[i-1] * (period - 1) + down[i-1]) / period
    
    rs = np.where(avg_down != 0, avg_up / avg_down, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_divergence(price, indicator, lookback=5):
    """Find bullish/bearish divergence between price and indicator."""
    bullish_div = np.zeros_like(price, dtype=bool)
    bearish_div = np.zeros_like(price, dtype=bool)
    
    for i in range(lookback, len(price)):
        # Bullish divergence: price makes lower low, indicator makes higher low
        if (price[i] < price[i-lookback] and 
            indicator[i] > indicator[i-lookback]):
            # Check if this is a significant low point
            if i >= lookback*2:
                price_low = np.min(price[i-lookback*2:i-lookback])
                ind_low = np.min(indicator[i-lookback*2:i-lookback])
                if price[i] <= price_low and indicator[i] >= ind_low:
                    bullish_div[i] = True
        
        # Bearish divergence: price makes higher high, indicator makes lower high
        if (price[i] > price[i-lookback] and 
            indicator[i] < indicator[i-lookback]):
            # Check if this is a significant high point
            if i >= lookback*2:
                price_high = np.max(price[i-lookback*2:i-lookback])
                ind_high = np.max(indicator[i-lookback*2:i-lookback])
                if price[i] >= price_high and indicator[i] <= ind_high:
                    bearish_div[i] = True
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Get weekly data for RSI calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:  # Need sufficient weekly data
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    
    # Weekly RSI (14-period)
    weekly_rsi = calculate_rsi(weekly_close, 14)
    
    # Align weekly RSI to daily timeframe
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rsi)
    
    # Find divergences on weekly RSI
    bullish_div, bearish_div = find_divergence(weekly_close, weekly_rsi, lookback=3)
    
    # Align divergence signals to daily timeframe
    bullish_div_aligned = align_htf_to_ltf(prices, df_weekly, bullish_div.astype(float))
    bearish_div_aligned = align_htf_to_ltf(prices, df_weekly, bearish_div.astype(float))
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_rsi_aligned[i]) or 
            np.isnan(bullish_div_aligned[i]) or 
            np.isnan(bearish_div_aligned[i]) or
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Convert divergence to boolean (handle any floating point issues)
        bullish_signal = bool(bullish_div_aligned[i] > 0.5)
        bearish_signal = bool(bearish_div_aligned[i] > 0.5)
        
        if position == 1:  # Long position
            # Exit if bearish divergence appears or price breaks below VWAP
            if bearish_signal or close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if bullish divergence appears or price breaks above VWAP
            if bullish_signal or close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish weekly RSI divergence and price above daily VWAP
            if bullish_signal and close[i] > vwap[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish weekly RSI divergence and price below daily VWAP
            elif bearish_signal and close[i] < vwap[i]:
                position = -1
                signals[i] = -0.25
    
    return signals