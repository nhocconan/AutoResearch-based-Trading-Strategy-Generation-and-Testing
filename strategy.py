#!/usr/bin/env python3
"""
4h_rsi_divergence_1d_trend_volume_v1
Hypothesis: On 4h timeframe, combine daily trend filter with RSI divergence signals for high-probability reversals. Uses daily EMA50/EMA200 trend filter, RSI(14) with bearish/bullish divergence detection, and volume confirmation. Enters short on bearish divergence (price higher high, RSI lower high) in downtrend; enters long on bullish divergence (price lower low, RSI higher low) in uptrend. Exits on opposite divergence or trend reversal. Designed for 15-30 trades/year with strict entry conditions to minimize fee drag while capturing major reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_divergence_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate daily EMA50 and EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    
    # Align to 4h timeframe (shifted by 1 day)
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation (20-period average on 4h = ~1.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Track recent highs/lows for divergence detection
    lookback = 10  # Look back 10 periods (~2.5 days) for swing points
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_4h[i]) or np.isnan(ema200_1d_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or i < lookback):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Determine trend based on daily EMA50 vs EMA200
        uptrend = ema50_1d_4h[i] > ema200_1d_4h[i]
        downtrend = ema50_1d_4h[i] < ema200_1d_4h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on bearish divergence (potential reversal)
            if downtrend:  # Only check for bearish div in downtrend context
                # Find recent price high and RSI high
                price_high_idx = np.argmax(high[i-lookback:i+1]) + i - lookback
                rsi_high_idx = np.argmax(rsi[i-lookback:i+1]) + i - lookback
                if (price_high_idx == i and rsi_high_idx < i and  # Current bar is price high
                    rsi[i] < rsi[rsi_high_idx]):  # RSI lower than previous high
                    exit_long = True
            # Exit on trend reversal to downtrend
            elif not uptrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on bullish divergence (potential reversal)
            if uptrend:  # Only check for bullish div in uptrend context
                # Find recent price low and RSI low
                price_low_idx = np.argmin(low[i-lookback:i+1]) + i - lookback
                rsi_low_idx = np.argmin(rsi[i-lookback:i+1]) + i - lookback
                if (price_low_idx == i and rsi_low_idx < i and  # Current bar is price low
                    rsi[i] > rsi[rsi_low_idx]):  # RSI higher than previous low
                    exit_short = True
            # Exit on trend reversal to uptrend
            elif not downtrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = False
            if uptrend or (not downtrend and not uptrend):  # Allow in uptrend or sideways
                # Find two most recent price lows
                window = high[i-lookback*2:i+1]  # Look further back for two points
                if len(window) >= lookback*2:
                    price_lows = []
                    rsi_lows = []
                    # Find local minima in price
                    for j in range(i-lookback*2, i+1):
                        if j >= lookback and j < n:
                            is_low = True
                            for k in range(max(lookback, j-5), min(n, j+6)):
                                if k != j and low[k] <= low[j]:
                                    is_low = False
                                    break
                            if is_low:
                                price_lows.append((j, low[j]))
                                rsi_lows.append((j, rsi[j]))
                    
                    # Need at least two lows
                    if len(price_lows) >= 2:
                        # Get two most recent lows
                        (price_low1, rsi_low1) = price_lows[-2]
                        (price_low2, rsi_low2) = price_lows[-1]
                        # Bullish div: lower price low, higher RSI low
                        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                            bullish_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = False
            if downtrend or (not downtrend and not uptrend):  # Allow in downtrend or sideways
                # Find two most recent price highs
                price_highs = []
                rsi_highs = []
                # Find local maxima in price
                for j in range(i-lookback*2, i+1):
                    if j >= lookback and j < n:
                        is_high = True
                        for k in range(max(lookback, j-5), min(n, j+6)):
                            if k != j and high[k] >= high[j]:
                                is_high = False
                                break
                        if is_high:
                            price_highs.append((j, high[j]))
                            rsi_highs.append((j, rsi[j]))
                
                # Need at least two highs
                if len(price_highs) >= 2:
                    # Get two most recent highs
                    (price_high1, rsi_high1) = price_highs[-2]
                    (price_high2, rsi_high2) = price_highs[-1]
                    # Bearish div: higher price high, lower RSI high
                    if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                        bearish_div = True
            
            # Volume confirmation required for entry
            if bullish_div and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif bearish_div and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals