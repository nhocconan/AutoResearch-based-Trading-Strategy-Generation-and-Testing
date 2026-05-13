#!/usr/bin/env python3
# 1d_1w_RSI_Pullback_Trend_Follower
# Hypothesis: Capture multi-week trends in BTC/ETH by buying pullbacks to weekly VWAP during strong weekly uptrends (and vice versa for shorts), using daily RSI for entry timing and volume confirmation to avoid false signals. Weekly trend filter prevents counter-trend trades. Designed to work in both bull and bear markets by following the dominant weekly trend.

name = "1d_1w_RSI_Pullback_Trend_Follower"
timeframe = "1d"
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

    # Get weekly data for trend filter and VWAP
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Weekly VWAP (volume-weighted average price)
    vwap_1w = (np.cumsum(volume_1w * (high_1w + low_1w + close_1w) / 3) / np.cumsum(volume_1w))
    vwap_1w = np.where(np.cumsum(volume_1w) == 0, np.nan, vwap_1w)

    # Align weekly indicators to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)

    # Daily RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Daily volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma  # 50% above average

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine weekly trend
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]

        # Price relative to weekly VWAP
        price_near_vwap = abs(close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i] < 0.02  # Within 2% of VWAP

        if position == 0:
            # LONG: Weekly uptrend + price near weekly VWAP (pullback) + RSI < 40 (not overbought) + volume surge
            if weekly_uptrend and price_near_vwap and rsi[i] < 40 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + price near weekly VWAP (pullback) + RSI > 60 (not oversold) + volume surge
            elif weekly_downtrend and price_near_vwap and rsi[i] > 60 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly trend turns down OR price moves 2% above VWAP (take profit) OR RSI > 70 (overbought)
            if not weekly_uptrend or close[i] > vwap_1w_aligned[i] * 1.02 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend turns up OR price moves 2% below VWAP (take profit) OR RSI < 30 (oversold)
            if not weekly_downtrend or close[i] < vwap_1w_aligned[i] * 0.98 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals