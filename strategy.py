#!/usr/bin/env python3
# 6h_OrderFlow_Imbalance_12hTrend
# Hypothesis: Use order flow imbalance (buying/selling pressure) from volume delta and 12h trend filter.
# Long when buying pressure > selling pressure with volume confirmation and 12h EMA50 uptrend.
# Short when selling pressure > buying pressure with volume confirmation and 12h EMA50 downtrend.
# Exit on mean reversion to 12h VWAP. Designed for low turnover (~20-30/year) to avoid fee drag.

name = "6h_OrderFlow_Imbalance_12hTrend"
timeframe = "6h"
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
    taker_buy_volume = prices['taker_buy_volume'].values

    # Calculate buying and selling pressure
    # Buying pressure = taker buy volume
    # Selling pressure = volume - taker buy volume
    buying_pressure = taker_buy_volume
    selling_pressure = volume - taker_buy_volume
    
    # Order flow imbalance: (buying - selling) / volume
    # Avoid division by zero
    volume_safe = np.where(volume == 0, 1, volume)
    ofi = (buying_pressure - selling_pressure) / volume_safe
    
    # Smooth OFI with 5-period EMA to reduce noise
    ofi_series = pd.Series(ofi)
    ofi_ema = ofi_series.ewm(span=5, adjust=False, min_periods=5).values

    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)

    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Get 12h VWAP for exit (mean reversion target)
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    vwap_12h = (typical_price_12h * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h_values = vwap_12h.values
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h_values)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(ofi_ema[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vwap_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: buying pressure > selling pressure with volume spike and 12h EMA50 uptrend
            if ofi_ema[i] > 0.1 and volume_spike[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: selling pressure > buying pressure with volume spike and 12h EMA50 downtrend
            elif ofi_ema[i] < -0.1 and volume_spike[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below 12h VWAP (mean reversion)
            if close[i] < vwap_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above 12h VWAP
            if close[i] > vwap_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals