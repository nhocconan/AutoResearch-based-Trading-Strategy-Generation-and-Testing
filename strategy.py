#!/usr/bin/env python3
# 4h_RSI_Divergence_BullBear
# Hypothesis: Use 4-hour RSI divergence with price action for high-probability reversals.
# Bullish divergence: price makes lower low, RSI makes higher low → long.
# Bearish divergence: price makes higher high, RSI makes lower high → short.
# Filtered by 1-day EMA50 trend and volume confirmation to avoid counter-trend trades.
# Designed for low trade frequency (<50/year) with high win rate in both bull and bear markets.

name = "4h_RSI_Divergence_BullBear"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for RSI calculation (same timeframe but we need it for divergence)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate RSI(14) on 4h closes
    delta = pd.Series(df_4h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Need sufficient lookback for divergence
        # Skip if any required value is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(rsi_values[i-1]) or np.isnan(rsi_values[i-2]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Bullish divergence: price lower low, RSI higher low
            bull_div = (low[i] < low[i-2] and 
                       low[i-1] > low[i-2] and  # Ensure we're checking a proper swing low
                       rsi_values[i] > rsi_values[i-2])
            
            # Bearish divergence: price higher high, RSI lower high
            bear_div = (high[i] > high[i-2] and 
                       high[i-1] < high[i-2] and  # Ensure we're checking a proper swing high
                       rsi_values[i] < rsi_values[i-2])
            
            if bull_div and close[i] > ema_50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.3:
                signals[i] = 0.25
                position = 1
            elif bear_div and close[i] < ema_50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below swing low or RSI overbought
            if (low[i] < low[i-1] or rsi_values[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above swing high or RSI oversold
            if (high[i] > high[i-1] or rsi_values[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals