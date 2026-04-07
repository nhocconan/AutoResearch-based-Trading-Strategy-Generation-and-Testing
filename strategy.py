#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Weekly KAMA Trend with Volume and Momentum Filter
# Hypothesis: KAMA trend direction from weekly data provides reliable trend filter
# for 4h entries. Enter long when price > KAMA and RSI > 50 with volume confirmation
# in bull markets; enter short when price < KAMA and RSI < 50 with volume confirmation
# in bear markets. Uses momentum to avoid whipsaws. Target: 20-40 trades/year.

name = "4h_weekly_kama_trend_volume_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    close_weekly = df_weekly['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_weekly, prepend=close_weekly[0]))
    volatility = np.abs(np.diff(close_weekly))
    er = np.zeros_like(close_weekly)
    for i in range(1, len(close_weekly)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_weekly)
    kama[0] = close_weekly[0]
    for i in range(1, len(close_weekly)):
        kama[i] = kama[i-1] + sc[i] * (close_weekly[i] - kama[i-1])
    
    # Shift by 1 to use previous week's data
    kama = np.roll(kama, 1)
    kama[0] = kama[1] if len(kama) > 1 else close_weekly[0]
    
    # Align to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_weekly, kama)
    
    # RSI filter on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < KAMA or RSI < 40 or volume filter fails
            if close[i] < kama_aligned[i] or rsi[i] < 40 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price > KAMA or RSI > 60 or volume filter fails
            if close[i] > kama_aligned[i] or rsi[i] > 60 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price > KAMA and RSI > 50 with volume
            if close[i] > kama_aligned[i] and rsi[i] > 50 and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price < KAMA and RSI < 50 with volume
            elif close[i] < kama_aligned[i] and rsi[i] < 50 and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals