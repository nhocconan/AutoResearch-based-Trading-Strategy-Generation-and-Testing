#!/usr/bin/env python3
# 12h_VWAP_StdDev_Breakout_1dTrend_Volume
# Hypothesis: Use VWAP with standard deviation bands as dynamic support/resistance on 12h timeframe.
# Enter long when price breaks above VWAP + 2*std with 1d EMA uptrend and volume spike.
# Enter short when price breaks below VWAP - 2*std with 1d EMA downtrend and volume spike.
# Exit when price returns to VWAP (mean reversion to fair value).
# VWAP adapts to market conditions, providing dynamic levels that work in both trending and ranging markets.
# Combined with 1d trend filter and volume confirmation to reduce false signals.
# Target: 15-30 trades/year on 12h to minimize fee drag while capturing strong moves.

name = "12h_VWAP_StdDev_Breakout_1dTrend_Volume"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate VWAP and standard deviation bands
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Calculate rolling standard deviation of price deviations from VWAP
    price_dev = typical_price - vwap
    # Use 24-period window (2 days of 12h data) for stability
    vwap_std = np.sqrt(np.convolve(price_dev**2, np.ones(24)/24, mode='same'))
    # Handle edges - use expanding window for first values
    for i in range(24):
        if i == 0:
            vwap_std[i] = price_dev[i]**2
        else:
            vwap_std[i] = np.sqrt(np.mean(price_dev[max(0, i-23):i+1]**2))
    vwap_std = np.where(np.isnan(vwap_std), 0, vwap_std)

    vwap_upper = vwap + 2.0 * vwap_std
    vwap_lower = vwap - 2.0 * vwap_std

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2.0x 24-period average
    vol_avg_24 = np.convolve(volume, np.ones(24)/24, mode='same')
    # Handle edges
    for i in range(len(volume)):
        if i < 24:
            vol_avg_24[i] = np.mean(volume[:i+1]) if i >= 0 else 0
        else:
            vol_avg_24[i] = np.mean(volume[i-23:i+1])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN or invalid
        if (np.isnan(vwap[i]) or np.isnan(vwap_upper[i]) or np.isnan(vwap_lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i]) or vol_avg_24[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above VWAP + 2*std + price > 1d EMA34 + volume spike
            if (close[i] > vwap_upper[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below VWAP - 2*std + price < 1d EMA34 + volume spike
            elif (close[i] < vwap_lower[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below VWAP (mean reversion to fair value)
            if close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above VWAP
            if close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals