#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour EMA trend with 4-hour RSI pullback entries
# Long when 12h EMA > 12h EMA previous and 4h RSI < 30 (pullback in uptrend)
# Short when 12h EMA < 12h EMA previous and 4h RSI > 70 (pullback in downtrend)
# Uses 12h trend filter to avoid counter-trend trades, RSI for entry timing during pullbacks
# Volume confirmation ensures institutional participation. Works in both bull/bear markets
# by trading with the higher timeframe trend. Target: 20-40 trades per year (80-160 over 4 years).

name = "4h_12hEMA_RSIPullback_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA trend ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # 12h EMA(21) for trend direction
    ema_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(100).values  # Fill initial NaN with 100
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(rsi_values[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h EMA trending up AND 4h RSI oversold (<30) with volume
            if ema_12h_aligned[i] > ema_12h_aligned[i-1] and rsi_values[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: 12h EMA trending down AND 4h RSI overbought (>70) with volume
            elif ema_12h_aligned[i] < ema_12h_aligned[i-1] and rsi_values[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 12h EMA trend turns down OR RSI overbought (>70)
            if ema_12h_aligned[i] < ema_12h_aligned[i-1] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 12h EMA trend turns up OR RSI oversold (<30)
            if ema_12h_aligned[i] > ema_12h_aligned[i-1] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals