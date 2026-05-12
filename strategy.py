#!/usr/bin/env python3
"""
12h_Volume_Weighted_RSI_Trend_Filter
Hypothesis: Combines RSI momentum with volume-weighted price action on 12h timeframe to capture sustained trends while avoiding whipsaws. Uses volume-weighted RSI (VW-RSI) to identify overextended conditions, with trend confirmation from 1d EMA50. Volume confirmation ensures institutional participation. Designed for 12-30 trades/year to minimize fee drag while working in both bull (trend continuation) and bear (mean reversion at extremes) regimes.
"""

name = "12h_Volume_Weighted_RSI_Trend_Filter"
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

    # Get 1d data for trend context (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 12-period RSI on close prices
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data

    # Calculate volume-weighted RSI (VW-RSI)
    # Weight RSI by volume to emphasize moves with participation
    vol_weight = volume / (np.mean(volume) + 1e-10)  # Avoid division by zero
    vw_rsi = rsi * vol_weight
    vw_rsi = np.clip(vw_rsi, 0, 100)  # Keep in RSI bounds

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        vw_rsi_val = vw_rsi[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma = np.mean(volume[max(0, i-20):i+1])  # 20-period volume average

        if np.isnan(vw_rsi_val) or np.isnan(ema50_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VW-RSI > 60 (bullish momentum with volume) + price above 1d EMA50 (uptrend)
            if vw_rsi_val > 60 and close[i] > ema50_val:
                signals[i] = 0.25
                position = 1
            # SHORT: VW-RSI < 40 (bearish momentum with volume) + price below 1d EMA50 (downtrend)
            elif vw_rsi_val < 40 and close[i] < ema50_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VW-RSI < 50 (loss of bullish momentum) or trend reversal
            if vw_rsi_val < 50 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VW-RSI > 50 (loss of bearish momentum) or trend reversal
            if vw_rsi_val > 50 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals