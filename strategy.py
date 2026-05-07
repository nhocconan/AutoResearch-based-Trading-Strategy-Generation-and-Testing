#!/usr/bin/env python3
# 1h_1d_RSI_Divergence_4hTrend
# Hypothesis: Combines 1d RSI divergence for reversal signals with 4h EMA trend filter and volume confirmation.
# Uses 1h for entry timing to reduce lag. Designed to work in both bull and bear markets by fading extremes.
# Target: 15-30 trades/year to minimize fee drag while capturing high-probability reversals.

name = "1h_1d_RSI_Divergence_4hTrend"
timeframe = "1h"
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
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume spike filter on 1h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # RSI divergence detection (bullish: price lower low, RSI higher low)
    # Bearish: price higher high, RSI lower high
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Track recent extremes for divergence
    lookback = 10
    price_low = np.full(n, np.nan)
    price_high = np.full(n, np.nan)
    rsi_low = np.full(n, np.nan)
    rsi_high = np.full(n, np.nan)
    
    for i in range(lookback, n):
        price_low[i] = np.min(low[i-lookback:i+1])
        price_high[i] = np.max(high[i-lookback:i+1])
        if not np.isnan(rsi_1d_aligned[i-lookback:i+1]).all():
            rsi_low[i] = np.nanmin(rsi_1d_aligned[i-lookback:i+1])
            rsi_high[i] = np.nanmax(rsi_1d_aligned[i-lookback:i+1])
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(price_low[i]) or np.isnan(rsi_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] <= price_low[i] * 1.001 and  # Allow small tolerance
                        rsi_1d_aligned[i] > rsi_low[i] and
                        rsi_1d_aligned[i] < 40)  # Oversold threshold
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] >= price_high[i] * 0.999 and  # Allow small tolerance
                        rsi_1d_aligned[i] < rsi_high[i] and
                        rsi_1d_aligned[i] > 60)  # Overbought threshold
            
            # Trend filter: price above/below 4h EMA20
            uptrend = close[i] > ema_20_4h_aligned[i]
            downtrend = close[i] < ema_20_4h_aligned[i]
            
            if bull_div and downtrend and volume_spike[i]:
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            elif bear_div and uptrend and volume_spike[i]:
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: RSI returns to neutral or trend changes
            if (rsi_1d_aligned[i] >= 50 or 
                close[i] < ema_20_4h_aligned[i] or
                bars_since_entry >= 10):  # Max hold period
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI returns to neutral or trend changes
            if (rsi_1d_aligned[i] <= 50 or 
                close[i] > ema_20_4h_aligned[i] or
                bars_since_entry >= 10):  # Max hold period
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
    
    return signals