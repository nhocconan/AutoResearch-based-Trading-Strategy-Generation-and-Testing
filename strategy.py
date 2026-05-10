#!/usr/bin/env python3
"""
4H_RSI_Divergence_With_Volume_Confirmation
Hypothesis: Uses bearish/bullish RSI divergence (price makes higher high/low, RSI makes lower high/higher low) 
combined with volume spike and 1d EMA trend filter to capture reversals in both bull and bear markets.
Designed for 4h timeframe with low trade frequency (<30/year) to avoid fee drag. 
Uses discrete position sizing (0.25) and exits on RSI mean reversion or trend failure.
"""

name = "4H_RSI_Divergence_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 4h chart
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate RSI peaks and troughs for divergence detection
    rsi_peaks = np.full(n, np.nan)
    rsi_troughs = np.full(n, np.nan)
    price_peaks = np.full(n, np.nan)
    price_troughs = np.full(n, np.nan)
    
    # Find local peaks and troughs in RSI (3-bar window)
    for i in range(2, n-2):
        if rsi_values[i] > rsi_values[i-1] and rsi_values[i] > rsi_values[i+1] and \
           rsi_values[i] > rsi_values[i-2] and rsi_values[i] > rsi_values[i+2]:
            rsi_peaks[i] = rsi_values[i]
            price_peaks[i] = close[i]
        if rsi_values[i] < rsi_values[i-1] and rsi_values[i] < rsi_values[i+1] and \
           rsi_values[i] < rsi_values[i-2] and rsi_values[i] < rsi_values[i+2]:
            rsi_troughs[i] = rsi_values[i]
            price_troughs[i] = close[i]
    
    # Forward fill the last peak/trough values
    last_rsi_peak = np.full(n, np.nan)
    last_price_peak = np.full(n, np.nan)
    last_rsi_trough = np.full(n, np.nan)
    last_price_trough = np.full(n, np.nan)
    
    peak_val = np.nan
    peak_price = np.nan
    trough_val = np.nan
    trough_price = np.nan
    
    for i in range(n):
        if not np.isnan(rsi_peaks[i]):
            peak_val = rsi_peaks[i]
            peak_price = price_peaks[i]
        last_rsi_peak[i] = peak_val
        last_price_peak[i] = peak_price
        
        if not np.isnan(rsi_troughs[i]):
            trough_val = rsi_troughs[i]
            trough_price = price_troughs[i]
        last_rsi_trough[i] = trough_val
        last_price_trough[i] = trough_price
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_threshold[i]) or \
           np.isnan(last_rsi_peak[i]) or np.isnan(last_price_peak[i]) or \
           np.isnan(last_rsi_trough[i]) or np.isnan(last_price_trough[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Bullish divergence: price makes higher low, RSI makes lower low
            bullish_div = (close[i] > last_price_trough[i] and 
                           rsi_values[i] < last_rsi_trough[i] and
                           last_price_trough[i] > 0 and last_rsi_trough[i] > 0)
            
            # Bearish divergence: price makes lower high, RSI makes higher high
            bearish_div = (close[i] < last_price_peak[i] and 
                           rsi_values[i] > last_rsi_peak[i] and
                           last_price_peak[i] > 0 and last_rsi_peak[i] > 0)
            
            # Long entry: bullish divergence + above EMA + volume spike
            if bullish_div and price_above_ema and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish divergence + below EMA + volume spike
            elif bearish_div and price_below_ema and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to 50 (mean reversion) or trend fails
            if rsi_values[i] >= 50 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to 50 (mean reversion) or trend fails
            if rsi_values[i] <= 50 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals