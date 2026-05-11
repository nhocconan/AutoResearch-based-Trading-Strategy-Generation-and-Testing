#!/usr/bin/env python3
"""
4h_RSI_Divergence_with_Volume_v1
Hypothesis: Combines RSI divergence detection with volume confirmation and trend filter.
In bull markets, bullish RSI divergence during pullbacks signals long entries.
In bear markets, bearish RSI divergence during rallies signals short entries.
Uses 1d EMA for trend filter and volume spike for confirmation.
Target: 80-160 trades over 4 years (20-40/year) on 4h timeframe.
"""

name = "4h_RSI_Divergence_with_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === RSI Calculation (14-period) ===
    def compute_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = compute_rsi(close, 14)
    
    # === Volume Spike Detection (20-period average) ===
    def compute_volume_spike(vol, period=20):
        vol_ma = np.zeros_like(vol)
        for i in range(len(vol)):
            if i < period:
                vol_ma[i] = np.mean(vol[max(0, i-period+1):i+1])
            else:
                vol_ma[i] = np.mean(vol[i-period+1:i+1])
        # Spike when current volume > 1.5 * moving average
        spike = vol > (1.5 * vol_ma)
        return spike
    
    vol_spike = compute_volume_spike(volume, 20)
    
    # === RSI Divergence Detection ===
    def find_divergences(price, rsi, lookback=10):
        bullish_div = np.full(len(price), False)
        bearish_div = np.full(len(price), False)
        
        for i in range(lookback, len(price)):
            # Look for bullish divergence: price makes lower low, RSI makes higher low
            if i >= lookback:
                # Find recent price low and RSI low
                price_window = price[i-lookback:i+1]
                rsi_window = rsi[i-lookback:i+1]
                
                # Find local minima in price and RSI
                price_min_idx = np.argmin(price_window)
                rsi_min_idx = np.argmin(rsi_window)
                
                # Bullish divergence: price lower low but RSI higher low
                if (price_min_idx == lookback and  # recent price low
                    price[i] < price[i-lookback] and  # current price < price lookback ago
                    rsi[i] > rsi[i-lookback]):      # current RSI > RSI lookback ago
                    bullish_div[i] = True
                
                # Bearish divergence: price higher high but RSI lower high
                price_max_idx = np.argmax(price_window)
                rsi_max_idx = np.argmax(rsi_window)
                if (price_max_idx == lookback and  # recent price high
                    price[i] > price[i-lookback] and  # current price > price lookback ago
                    rsi[i] < rsi[i-lookback]):      # current RSI < RSI lookback ago
                    bearish_div[i] = True
        
        return bullish_div, bearish_div
    
    bullish_div, bearish_div = find_divergences(close, rsi, 10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(bullish_div[i]) or 
            np.isnan(bearish_div[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish RSI divergence + volume spike + in uptrend (price > EMA50)
            if bullish_div[i] and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence + volume spike + in downtrend (price < EMA50)
            elif bearish_div[i] and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish divergence or price breaks below EMA50
            if bearish_div[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: bullish divergence or price breaks above EMA50
            if bullish_div[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals