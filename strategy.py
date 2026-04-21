#!/usr/bin/env python3
"""
4h_12h_RSI_Divergence_Volume_Confirmation
Hypothesis: Use RSI divergences on 12h timeframe combined with volume confirmation on 4h to capture reversals.
Long when bullish RSI divergence forms on 12h (price makes lower low, RSI makes higher low) with volume confirmation.
Short when bearish RSI divergence forms on 12h (price makes higher high, RSI makes lower high) with volume confirmation.
Exit when RSI crosses above 50 (for longs) or below 50 (for shorts).
Uses volume confirmation to filter false signals and reduce overtrading. Designed for 4h to limit trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for RSI divergence calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate RSI on 12h (14-period)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate volume confirmation on 4h
    volume = prices['volume'].values
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:10] = np.nan  # Not enough data for MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track recent highs/lows for divergence detection
    lookback = 10  # Look back 10 periods for swing points
    
    for i in range(30, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_ok = vol_ratio > 1.5  # Volume > 1.5x average
        
        if position == 0:
            # Check for bullish divergence: price lower low, RSI higher low
            bullish_div = False
            if i >= lookback:
                # Find recent swing low in price
                price_lows = []
                for j in range(i-lookback, i+1):
                    if j == 0 or prices['low'].iloc[j] < prices['low'].iloc[j-1]:
                        price_lows.append(j)
                if len(price_lows) >= 2:
                    low1_idx, low2_idx = price_lows[-2], price_lows[-1]
                    price_low1 = prices['low'].iloc[low1_idx]
                    price_low2 = prices['low'].iloc[low2_idx]
                    rsi_low1 = rsi_12h_aligned[low1_idx]
                    rsi_low2 = rsi_12h_aligned[low2_idx]
                    if not (np.isnan(rsi_low1) or np.isnan(rsi_low2)):
                        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                            bullish_div = True
            
            # Check for bearish divergence: price higher high, RSI lower high
            bearish_div = False
            if i >= lookback:
                # Find recent swing high in price
                price_highs = []
                for j in range(i-lookback, i+1):
                    if j == 0 or prices['high'].iloc[j] > prices['high'].iloc[j-1]:
                        price_highs.append(j)
                if len(price_highs) >= 2:
                    high1_idx, high2_idx = price_highs[-2], price_highs[-1]
                    price_high1 = prices['high'].iloc[high1_idx]
                    price_high2 = prices['high'].iloc[high2_idx]
                    rsi_high1 = rsi_12h_aligned[high1_idx]
                    rsi_high2 = rsi_12h_aligned[high2_idx]
                    if not (np.isnan(rsi_high1) or np.isnan(rsi_high2)):
                        if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                            bearish_div = True
            
            # Enter long on bullish divergence with volume confirmation
            if bullish_div and volume_ok:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish divergence with volume confirmation
            elif bearish_div and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when RSI crosses above 50 (momentum fading)
            if rsi_12h_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI crosses below 50 (momentum fading)
            if rsi_12h_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_RSI_Divergence_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0