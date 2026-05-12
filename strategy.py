#!/usr/bin/env python3
"""
12h_1w_Trend_1d_Pullback_Swing
Hypothesis: Buy pullbacks to 1w VWAP in a rising 1w trend, sell rallies to 1w VWAP in a falling 1w trend. Uses 1d RSI to time entries on 12h timeframe. Designed for low trade frequency (20-40/year) with strong trend capture and mean-reversion entries within the trend. Works in bull markets via long pullbacks and in bear markets via short rallies.
Timeframe: 12h
"""

name = "12h_1w_Trend_1d_Pullback_Swing"
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

    # Get weekly data for trend and VWAP ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Weekly VWAP (volume-weighted average price)
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vp_1w = typical_price_1w * df_1w['volume'].values
    cum_vp_1w = np.cumsum(vp_1w)
    cum_vol_1w = np.cumsum(df_1w['volume'].values)
    vwap_1w = np.divide(cum_vp_1w, cum_vol_1w, out=np.full_like(cum_vp_1w, np.nan), where=cum_vol_1w!=0)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)

    # Get daily data for RSI pullback filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)

    # Daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after warmup
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 1w trend up (close > EMA20) + pullback to VWAP (close <= VWAP) + RSI not overbought (<60)
            if (close[i] > ema_20_1w_aligned[i] and 
                close[i] <= vwap_1w_aligned[i] and 
                rsi_1d_aligned[i] < 60):
                signals[i] = 0.25
                position = 1
            # SHORT: 1w trend down (close < EMA20) + rally to VWAP (close >= VWAP) + RSI not oversold (>40)
            elif (close[i] < ema_20_1w_aligned[i] and 
                  close[i] >= vwap_1w_aligned[i] and 
                  rsi_1d_aligned[i] > 40):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close > 1w VWAP (mean reversion complete) or trend broken
            if close[i] > vwap_1w_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close < 1w VWAP (mean reversion complete) or trend broken
            if close[i] < vwap_1w_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals