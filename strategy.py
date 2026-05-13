#!/usr/bin/env python3
"""
1D_KAMA_RSI_Chop_Filter_v1
Hypothesis: Daily KAMA trend + RSI overbought/oversold + Choppiness regime filter.
- KAMA adapts to market noise, reducing whipsaws in chop.
- RSI extremes trigger mean reversion entries when KAMA confirms trend.
- Choppiness filter avoids trades in strong trends where mean reversion fails.
- Works in bull/bear by adapting to regime: mean revert in chop, avoid in trend.
- Target: 15-25 trades/year to minimize fee drag.
"""

name = "1D_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
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

    # --- KAMA ( Kaufman Adaptive Moving Average ) ---
    er = np.zeros(n)  # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility as sum of absolute changes over ER period
    er_period = 10
    fast_sc = 2 / (2 + 1)  # for EMA 2
    slow_sc = 2 / (30 + 1) # for EMA 30
    volatility_sum = np.zeros(n)
    for i in range(er_period, n):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    price_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(volatility_sum != 0, price_change / volatility_sum, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # --- RSI (14) ---
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    # --- Choppiness Index (14) ---
    chop_period = 14
    atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(chop_period, n):
        atr[i] = np.mean(tr[i-chop_period+1:i+1])
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(chop_period-1, n):
        max_high[i] = np.max(high[i-chop_period+1:i+1])
        min_low[i] = np.min(low[i-chop_period+1:i+1])
    chop = np.zeros(n)
    for i in range(chop_period-1, n):
        if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(np.sum(tr[i-chop_period+1:i+1]) / (max_high[i] - min_low[i])) / np.log10(chop_period)
        else:
            chop[i] = 50  # neutral

    # --- Weekly trend filter (1w EMA34) ---
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(max(30, chop_period-1), n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: Chop > 50 = choppy/mean-revert favorable
        if chop[i] > 50:
            if position == 0:
                # Long: RSI oversold + price > KAMA (bullish bias)
                if rsi[i] < 30 and close[i] > kama[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI overbought + price < KAMA (bearish bias)
                elif rsi[i] > 70 and close[i] < kama[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long: RSI overbought or price < KAMA
                if rsi[i] > 70 or close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI oversold or price > KAMA
                if rsi[i] < 30 or close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In strong trend (Chop < 50), stay aside or follow weekly trend
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals