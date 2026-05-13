#!/usr/bin/env python3
# 1d_KAMA_Trend_With_Volume_And_Chop_Filter
# Hypothesis: Kaufman's Adaptive Moving Average (KAMA) provides adaptive trend filtering,
# which works well in both trending and ranging markets when combined with volume confirmation
# and Choppiness Index regime filter to avoid false signals in choppy conditions.
# Uses KAMA crossover signals with volume > 1.5x 20-period average and Choppiness Index > 61.8 (ranging)
# or < 38.2 (trending) to adapt strategy to market regime.
# Target: 7-25 trades/year per symbol with disciplined risk control via trailing stop.

name = "1d_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "1d"
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

    # Get 1w data for trend filter and regime detection
    df_1w = get_htf_data(prices, '1w')

    # Calculate KAMA ( Kaufman's Adaptive Moving Average )
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Proper ER calculation: |net change| / sum(|abs changes|) over period
    er = np.zeros_like(close)
    for i in range(10, n):  # ER period = 10
        if i >= 10:
            net_change = np.abs(close[i] - close[i-10])
            total_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = net_change / total_change if total_change > 0 else 0
    # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    fastest_sc = 2 / (2 + 1)   # EMA(2)
    slowest_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after ER period
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)

    # Weekly trend filter: price vs weekly EMA34
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Choppiness Index for regime detection (14-period)
    chop = np.full_like(close, np.nan)
    for i in range(14, n):
        true_range = np.maximum(high[i] - low[i],
                               np.maximum(np.abs(high[i] - close[i-1]),
                                          np.abs(low[i] - close[i-1])))
        atr14 = np.sum(true_range[i-14:i+1]) / 14
        highest_high = np.max(high[i-14:i+1])
        lowest_low = np.min(low[i-14:i+1])
        if atr14 > 0:
            chop[i] = 100 * np.log10(highest_high - lowest_low) / np.log10(14) / (atr14 * 14)
        else:
            chop[i] = 50

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: Only trade in trending markets (CHOP < 38.2) or strong reversals in chop (CHOP > 61.8)
        is_trending = chop[i] < 38.2
        is_choppy = chop[i] > 61.8

        if position == 0:
            # LONG: Price above KAMA AND above weekly EMA34 in uptrend OR oversold bounce in chop
            if ((close[i] > kama_aligned[i] and 
                 close[i] > ema34_1w_aligned[i] and 
                 volume_spike[i] and 
                 is_trending) or
                (close[i] < kama_aligned[i] * 0.98 and  # Oversold condition
                 volume_spike[i] and 
                 is_choppy)):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA AND below weekly EMA34 in downtrend OR overbought bounce in chop
            elif ((close[i] < kama_aligned[i] and 
                   close[i] < ema34_1w_aligned[i] and 
                   volume_spike[i] and 
                   is_trending) or
                  (close[i] > kama_aligned[i] * 1.02 and  # Overbought condition
                   volume_spike[i] and 
                   is_choppy)):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR trend turns down OR chop increases significantly
            if (close[i] < kama_aligned[i] or 
                close[i] < ema34_1w_aligned[i] or
                chop[i] > 50):  # Exit if market becomes choppy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR trend turns up OR chop increases significantly
            if (close[i] > kama_aligned[i] or 
                close[i] > ema34_1w_aligned[i] or
                chop[i] > 50):  # Exit if market becomes choppy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals