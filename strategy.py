# 6h RSI Divergence with 1D Trend Filter and Volume Spike
# Hypothesis: RSI divergence signals exhaustion in price momentum. Combined with 1D trend direction,
# this captures reversals in both bull and bear markets. Volume spike confirms institutional interest.
# Target: 15-30 trades/year per symbol with high win rate via confluence.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and RSI divergence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1D EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        else:
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1D EMA to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate RSI on 6h data
    rsi = calculate_rsi(close, 14)
    
    # Volume filter: volume > 2.0x average (20-period)
    vol_ma_20 = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI(14) + volume MA(20) + 1D EMA(50)
    start_idx = max(14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        rsi_now = rsi[i]
        
        # Trend filter
        trend_up = price_now > ema_1d_aligned[i]
        trend_down = price_now < ema_1d_aligned[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        # RSI divergence detection (simplified: look for extreme RSI with price action)
        # Bullish divergence: RSI oversold (<30) and rising while price makes lower low
        # Bearish divergence: RSI overbought (>70) and falling while price makes higher high
        
        if position == 0:
            # Look for bullish setup: RSI oversold + price holding + volume spike
            if rsi_now < 30 and vol_filter and trend_up:
                # Additional confirmation: price above prior low
                if i >= 2 and price_now > low[i-1]:
                    signals[i] = size
                    position = 1
            # Look for bearish setup: RSI overbought + price rejecting + volume spike
            elif rsi_now > 70 and vol_filter and trend_down:
                # Additional confirmation: price below prior high
                if i >= 2 and price_now < high[i-1]:
                    signals[i] = -size
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought or trend breaks down
            if rsi_now > 70 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold or trend breaks up
            if rsi_now < 30 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_Divergence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0