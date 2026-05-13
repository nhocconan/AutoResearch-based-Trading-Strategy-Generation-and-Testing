#!/usr/bin/env python3
# 12h_RSI_Extremes_1dTrend_Volume
# Hypothesis: At 12h timeframe, RSI extremes (>80 or <20) indicate overextended moves likely to reverse.
# Enter counter-trend when RSI reaches extreme AND 1d trend opposes the extreme (e.g., RSI>80 in 1d downtrend = short).
# Volume confirmation ensures institutional participation in the reversal.
# Works in both bull and bear markets by fading extremes in the direction of higher timeframe trend.
# Target: 15-30 trades/year per symbol to minimize fee drag.

name = "12h_RSI_Extremes_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data for trend and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d RSI(14)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align 1d indicators to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12h RSI(14) for entry timing
    close_12h = pd.Series(close)
    delta_12h = close_12h.diff()
    gain_12h = delta_12h.clip(lower=0)
    loss_12h = -delta_12h.clip(upper=0)
    avg_gain_12h = gain_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_12h = loss_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_12h = avg_gain_12h / avg_loss_12h
    rsi_12h = 100 - (100 / (1 + rs_12h))
    rsi_12h = rsi_12h.values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 12h RSI < 20 (oversold) + 1d downtrend + volume spike
            if (rsi_12h[i] < 20 and 
                close[i] < ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 12h RSI > 80 (overbought) + 1d uptrend + volume spike
            elif (rsi_12h[i] > 80 and 
                  close[i] > ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 12h RSI > 60 or trend reversal
            if rsi_12h[i] > 60 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 12h RSI < 40 or trend reversal
            if rsi_12h[i] < 40 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals