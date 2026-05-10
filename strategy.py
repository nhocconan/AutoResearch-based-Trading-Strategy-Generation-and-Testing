#!/usr/bin/env python3
"""
12h_RSI_Divergence_Trend_Filter
Hypothesis: Combines RSI divergence detection on 12h timeframe with 1w trend filter and volume confirmation.
In bull markets: buy on bullish RSI divergence during pullbacks in uptrend.
In bear markets: sell on bearish RSI divergence during rallies in downtrend.
Uses 1w EMA50 for trend filter and 1d volume spike for confirmation to reduce false signals.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
"""

name = "12h_RSI_Divergence_Trend_Filter"
timeframe = "12h"
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
    
    # RSI on 12h with period 14
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # Wilder's smoothing
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Detect RSI divergence: bullish (price lower low, RSI higher low) and bearish (price higher high, RSI lower high)
    def find_divergences(high_prices, low_prices, rsi_values, lookback=10):
        bullish_div = np.zeros_like(rsi_values, dtype=bool)
        bearish_div = np.zeros_like(rsi_values, dtype=bool)
        
        for i in range(lookback, len(rsi_values)):
            # Look for swing low in price
            if low_prices[i] == np.min(low_prices[i-lookback:i+1]):
                # Check if this is a higher low in RSI compared to previous swing low
                for j in range(i-lookback, i):
                    if low_prices[j] == np.min(low_prices[j-lookback:j+1]) and j < i:
                        if low_prices[i] < low_prices[j] and rsi_values[i] > rsi_values[j]:
                            bullish_div[i] = True
                            break
            
            # Look for swing high in price
            if high_prices[i] == np.max(high_prices[i-lookback:i+1]):
                # Check if this is a lower high in RSI compared to previous swing high
                for j in range(i-lookback, i):
                    if high_prices[j] == np.max(high_prices[j-lookback:j+1]) and j < i:
                        if high_prices[i] > high_prices[j] and rsi_values[i] < rsi_values[j]:
                            bearish_div[i] = True
                            break
        return bullish_div, bearish_div
    
    bullish_div, bearish_div = find_divergences(high, low, rsi, 10)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d volume SMA20 for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure RSI and divergence detection have enough data
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 2.0  # 2x 12h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        if position == 0:
            # Long: Bullish RSI divergence in uptrend with volume confirmation
            if bullish_div[i] and close[i] > ema50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish RSI divergence in downtrend with volume confirmation
            elif bearish_div[i] and close[i] < ema50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bearish RSI divergence or trend reversal
            if bearish_div[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bullish RSI divergence or trend reversal
            if bullish_div[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals