#!/usr/bin/env python3
# 1h_MultiTimeframe_Momentum_4hTrend
# Hypothesis: Use 4h momentum (RSI) for trend direction, 1h for precise entry timing with volume confirmation.
# Long when 4h RSI > 55 and rising, 1h price breaks above recent high with volume spike.
# Short when 4h RSI < 45 and falling, 1h price breaks below recent low with volume spike.
# Works in bull (momentum continuations) and bear (mean reversion from extremes via RSI).
# Low frequency due to 4h trend filter and volume confirmation requirement.

name = "1h_MultiTimeframe_Momentum_4hTrend"
timeframe = "1h"
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

    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    
    # 4h RSI for trend (14-period)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # 4h RSI slope (momentum direction)
    rsi_change = np.diff(rsi_4h, prepend=rsi_4h[0])
    rsi_slope = pd.Series(rsi_change).rolling(window=3, min_periods=1).mean().values
    
    # Align 4h indicators to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    rsi_slope_aligned = align_htf_to_ltf(prices, df_4h, rsi_slope)
    
    # 1h volume spike confirmation
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > 1.5 * vol_ma_10
    
    # 1h price action: recent high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(rsi_slope_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h bullish momentum + 1h breakout above resistance + volume
            if (rsi_4h_aligned[i] > 55 and 
                rsi_slope_aligned[i] > 0 and 
                close[i] > high_20[i-1] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h bearish momentum + 1h breakout below support + volume
            elif (rsi_4h_aligned[i] < 45 and 
                  rsi_slope_aligned[i] < 0 and 
                  close[i] < low_20[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h momentum weakening or 1h mean reversion
            if rsi_4h_aligned[i] < 50 or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h momentum weakening or 1h mean reversion
            if rsi_4h_aligned[i] > 50 or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals