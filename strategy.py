#!/usr/bin/env python3
# 1d_1w_rsi_divergence_volume_v1
# Strategy: Daily RSI divergence with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: RSI divergence (price makes new high/low but RSI does not) signals exhaustion.
# Bullish: bullish divergence (higher low in price, lower low in RSI) + price above weekly EMA50 + volume > 1.5x avg.
# Bearish: bearish divergence (lower high in price, higher high in RSI) + price below weekly EMA50 + volume > 1.5x avg.
# Works in bull markets by catching pullbacks and in bear markets by catching bounces.
# Uses tight entry conditions to limit trades (~15-30/year) and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_divergence_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Daily volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    # RSI divergence detection (requires 3-day lookback for swing points)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Bullish divergence: higher low in price, lower low in RSI
        if low[i] < low[i-1] and low[i-1] > low[i-2]:  # recent low at i-1
            if rsi[i] < rsi[i-1] and rsi[i-1] > rsi[i-2]:  # RSI making lower low
                # Check if current price low is higher than prior swing low
                j = i-2
                while j >= 0 and low[j] > low[i-1]:
                    j -= 1
                if j >= 0 and low[i] > low[j]:
                    bullish_div[i] = True
        
        # Bearish divergence: lower high in price, higher high in RSI
        if high[i] > high[i-1] and high[i-1] < high[i-2]:  # recent high at i-1
            if rsi[i] > rsi[i-1] and rsi[i-1] < rsi[i-2]:  # RSI making higher high
                # Check if current price high is lower than prior swing high
                j = i-2
                while j >= 0 and high[j] < high[i-1]:
                    j -= 1
                if j >= 0 and high[i] < high[j]:
                    bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if any required data is invalid
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: RSI divergence + volume + trend alignment
        if bullish_div[i] and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_div[i] and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite divergence with volume confirmation
        elif position == 1 and bearish_div[i] and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_div[i] and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals