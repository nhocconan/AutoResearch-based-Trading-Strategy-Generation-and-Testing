#!/usr/bin/env python3
# 6h_1d_rsi_divergence_volume_v1
# Hypothesis: 6h price RSI divergence with 1d trend filter and volume confirmation
# - Bullish divergence: price makes lower low, RSI makes higher low → long when 1d trend up
# - Bearish divergence: price makes higher high, RSI makes lower high → short when 1d trend down
# - Volume confirmation filters false signals
# Works in bull/bear markets by trading reversals against overextended moves
# Target: 20-50 trades/year

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_divergence_volume_v1"
timeframe = "6h"
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
    
    # Calculate RSI on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Lookback period for divergence detection
    lookback = 10
    
    for i in range(lookback, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns down
            if rsi[i] > 70 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns up
            if rsi[i] < 30 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for divergence
            # Need enough lookback
            if i < lookback:
                signals[i] = 0.0
                continue
            
            # Check for bullish divergence: price lower low, RSI higher low
            price_low_idx = i - np.argmin(low[i-lookback:i+1])
            rsi_low_idx = i - np.argmin(rsi[i-lookback:i+1])
            
            bullish_div = (low[price_low_idx] < low[i-lookback] and 
                          rsi[rsi_low_idx] > rsi[i-lookback] and
                          price_low_idx == rsi_low_idx)  # Same bar index
            
            # Check for bearish divergence: price higher high, RSI lower high
            price_high_idx = i - np.argmax(high[i-lookback:i+1])
            rsi_high_idx = i - np.argmax(rsi[i-lookback:i+1])
            
            bearish_div = (high[price_high_idx] > high[i-lookback] and 
                          rsi[rsi_high_idx] < rsi[i-lookback] and
                          price_high_idx == rsi_high_idx)  # Same bar index
            
            # Bullish entry: bullish divergence + 1d uptrend + volume
            if (bullish_div and 
                close[i] > ema_50_aligned[i] and  # 1d uptrend
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Bearish entry: bearish divergence + 1d downtrend + volume
            elif (bearish_div and 
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals