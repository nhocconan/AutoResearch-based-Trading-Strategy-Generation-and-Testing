#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Mean Reversion with 4h Trend Filter
# Long when: price < lower Bollinger Band (20,2.0) AND 4h EMA(21) rising
# Short when: price > upper Bollinger Band (20,2.0) AND 4h EMA(21) falling
# Exit when: price crosses middle Bollinger Band (SMA20) OR 4h EMA direction changes
# Uses 4h for trend direction (reduces whipsaw) and 1h for entry timing
# Bollinger Bands capture mean reversion in ranging markets, EMA filter avoids counter-trend trades
# Target: 60-150 trades over 4 years by requiring both Bollinger break and trend alignment
# Works in bull/bear: trend filter ensures we only trade with higher timeframe momentum

name = "1h_bb_meanrev_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2.0) on 1h
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    
    # 4h EMA(21) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 4h EMA direction (rising/falling)
    ema_rising = np.zeros(n, dtype=bool)
    ema_falling = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(ema_4h_aligned[i]) and not np.isnan(ema_4h_aligned[i-1]):
            ema_rising[i] = ema_4h_aligned[i] > ema_4h_aligned[i-1]
            ema_falling[i] = ema_4h_aligned[i] < ema_4h_aligned[i-1]
        else:
            ema_rising[i] = ema_rising[i-1]
            ema_falling[i] = ema_falling[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_4h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses middle band OR 4h EMA direction changes
        if position == 1:  # long position
            if close[i] > basis[i] or not ema_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] < basis[i] or not ema_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Bollinger break + 4h trend alignment
            # Long: price < lower band AND 4h EMA rising
            if close[i] < lower[i] and ema_rising[i]:
                signals[i] = 0.20
                position = 1
            # Short: price > upper band AND 4h EMA falling
            elif close[i] > upper[i] and ema_falling[i]:
                signals[i] = -0.20
                position = -1
    
    return signals