#!/usr/bin/env python3
# 4h_RSI_Pullback_With_Volume_Trend
# Hypothesis: Enter long on RSI pullback (30-40) during uptrend (price > EMA50) with volume confirmation; enter short on RSI bounce (60-70) during downtrend (price < EMA50) with volume confirmation.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades. RSI pullbacks in trending markets offer high-probability entries with favorable risk-reward.
# Volume surge confirms institutional interest. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Low frequency due to RSI range + trend + volume confluence.

name = "4h_RSI_Pullback_With_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI (14) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after RSI warmup
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI pullback (30-40) + uptrend + volume spike
            if 30 <= rsi[i] <= 40 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI bounce (60-70) + downtrend + volume spike
            elif 60 <= rsi[i] <= 70 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (momentum fade) OR trend reversal
            if rsi[i] > 50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (momentum fade) OR trend reversal
            if rsi[i] < 50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals