#!/usr/bin/env python3
# 1d_1w_momentum_divergence_v1
# Hypothesis: Trade momentum divergences between weekly RSI and price action on daily timeframe.
# Uses weekly RSI(14) to identify overbought/oversold conditions, with price making new highs/lows.
# In uptrend (price > weekly EMA50): look for bearish divergence (price makes higher high, RSI makes lower high) for short.
# In downtrend (price < weekly EMA50): look for bullish divergence (price makes lower low, RSI makes higher low) for long.
# Volume confirmation required: current volume > 1.5x 20-day average.
# Designed for low-frequency, high-conviction trades to avoid fee drag.
# Works in both bull and bear markets by trading reversals at extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_momentum_divergence_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for RSI and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA50 for trend filter
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track weekly highs/lows for divergence detection
    # We'll look for divergences over the last 3 weekly periods
    rsi_high = np.full_like(rsi, np.nan)
    rsi_low = np.full_like(rsi, np.nan)
    price_high = np.full_like(close_1w, np.nan)
    price_low = np.full_like(close_1w, np.nan)
    
    # Find weekly peaks and troughs
    for i in range(2, len(close_1w)):
        if (close_1w[i] > close_1w[i-1] and close_1w[i] > close_1w[i-2] and
            close_1w[i] > close_1w[i+1] if i+1 < len(close_1w) else True):
            price_high[i] = close_1w[i]
            rsi_high[i] = rsi[i]
        if (close_1w[i] < close_1w[i-1] and close_1w[i] < close_1w[i-2] and
            close_1w[i] < close_1w[i+1] if i+1 < len(close_1w) else True):
            price_low[i] = close_1w[i]
            rsi_low[i] = rsi[i]
    
    # Forward fill the peak/trough values
    price_high = pd.Series(price_high).ffill().bfill().values
    price_low = pd.Series(price_low).ffill().bfill().values
    rsi_high = pd.Series(rsi_high).ffill().bfill().values
    rsi_low = pd.Series(rsi_low).ffill().bfill().values
    
    # Align these to daily
    price_high_aligned = align_htf_to_ltf(prices, df_1w, price_high)
    price_low_aligned = align_htf_to_ltf(prices, df_1w, price_low)
    rsi_high_aligned = align_htf_to_ltf(prices, df_1w, rsi_high)
    rsi_low_aligned = align_htf_to_ltf(prices, df_1w, rsi_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(price_high_aligned[i]) or
            np.isnan(price_low_aligned[i]) or np.isnan(rsi_high_aligned[i]) or
            np.isnan(rsi_low_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly EMA50 or RSI becomes overbought (>70)
            if close[i] < ema50_aligned[i] or rsi_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly EMA50 or RSI becomes oversold (<30)
            if close[i] > ema50_aligned[i] or rsi_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Only in downtrend (price < weekly EMA50)
            bull_div = (close[i] < price_low_aligned[i] and 
                       rsi_aligned[i] > rsi_low_aligned[i] and
                       close[i] < ema50_aligned[i])
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            # Only in uptrend (price > weekly EMA50)
            bear_div = (close[i] > price_high_aligned[i] and 
                       rsi_aligned[i] < rsi_high_aligned[i] and
                       close[i] > ema50_aligned[i])
            
            # Long entry: bullish divergence with volume surge
            if bull_div and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish divergence with volume surge
            elif bear_div and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals