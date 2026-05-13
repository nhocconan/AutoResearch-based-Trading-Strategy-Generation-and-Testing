#!/usr/bin/env python3
# 1h_RSI_Reversal_With_Volume_Confirmation
# Hypothesis: Enter short when RSI > 70 and volume > 1.5x average during 4h downtrend; enter long when RSI < 30 and volume > 1.5x average during 4h uptrend.
# Uses 4h trend for direction, 1h for entry timing with RSI extremes and volume confirmation.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Low frequency due to RSI extremes and volume confirmation requirements.

name = "1h_RSI_Reversal_With_Volume_Confirmation"
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

    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after RSI warmup
        # Skip if any required value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 + volume confirmation + 4h uptrend
            if rsi[i] < 30 and volume_confirmed[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: RSI > 70 + volume confirmation + 4h downtrend
            elif rsi[i] > 70 and volume_confirmed[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 or trend reversal
            if rsi[i] > 50 or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI < 50 or trend reversal
            if rsi[i] < 50 or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals