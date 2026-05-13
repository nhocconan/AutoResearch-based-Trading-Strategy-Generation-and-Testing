#!/usr/bin/env python3
# 12h_RSI_Trend_Volume_Zone
# Hypothesis: Combine RSI mean reversion with trend alignment and volume confirmation on 12h.
# Long when RSI < 30 (oversold) + price above 200-period EMA (bullish bias) + volume spike.
# Short when RSI > 70 (overbought) + price below 200-period EMA (bearish bias) + volume spike.
# Uses weekly trend filter to avoid counter-trend trades in strong trends.
# Low frequency due to RSI extremes and volume confirmation requirements.

name = "12h_RSI_Trend_Volume_Zone"
timeframe = "12h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend: EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.finfo(float).eps, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # 200-period EMA for bias
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume spike: volume > 2.0 * 24-period average (2 days worth at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold + price above EMA200 + weekly uptrend + volume spike
            if rsi[i] < 30 and close[i] > ema200[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought + price below EMA200 + weekly downtrend + volume spike
            elif rsi[i] > 70 and close[i] < ema200[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought OR trend reversal
            if rsi[i] > 70 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold OR trend reversal
            if rsi[i] < 30 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals