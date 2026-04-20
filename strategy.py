#!/usr/bin/env python3
# 6h_1d_RSI_Divergence_With_Volume
# Hypothesis: On 6h timeframe, enter long on bullish RSI divergence (price makes lower low, RSI makes higher low) with volume confirmation and price above 6h EMA50.
# Enter short on bearish RSI divergence (price makes higher high, RSI makes lower high) with volume confirmation and price below 6h EMA50.
# Uses 1-day EMA50 as higher timeframe trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50.
# RSI divergence helps catch reversals in both bull and bear markets, while volume and EMA filters reduce false signals.
# Target: 15-30 trades per year.

name = "6h_1d_RSI_Divergence_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 6h RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 6h EMA50 for trend
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1-day EMA50 for higher timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Detect bullish RSI divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 2:
            # Look for lower low in price and higher low in RSI over last 3 periods
            if (low[i] < low[i-1] < low[i-2] and 
                rsi[i] > rsi[i-1] > rsi[i-2]):
                bullish_div = True
        
        # Detect bearish RSI divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 2:
            # Look for higher high in price and lower high in RSI over last 3 periods
            if (high[i] > high[i-1] > high[i-2] and 
                rsi[i] < rsi[i-1] < rsi[i-2]):
                bearish_div = True
        
        if position == 0:
            # Long: bullish divergence, volume spike, price above 6h EMA50, and price above 1d EMA50 (uptrend)
            if (bullish_div and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema_50[i] and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence, volume spike, price below 6h EMA50, and price below 1d EMA50 (downtrend)
            elif (bearish_div and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema_50[i] and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below 6h EMA50 or bearish divergence
            if close[i] < ema_50[i] or bearish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above 6h EMA50 or bullish divergence
            if close[i] > ema_50[i] or bullish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals