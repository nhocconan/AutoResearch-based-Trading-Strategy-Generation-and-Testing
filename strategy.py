#!/usr/bin/env python3
# 6h_ADX_Chaikin_OBOS_Strategy
# Hypothesis: Combine ADX (trend strength) with Chaikin Oscillator (momentum) and overbought/oversold levels to capture trend continuation in strong trends and mean reversion in weak trends. Uses 1d trend filter for higher timeframe alignment. Designed for low turnover (~20-30 trades/year) by requiring ADX > 25 for trend following and ADX < 20 for mean reversion, reducing whipsaws. Works in both bull and bear markets by adapting to regime.

name = "6h_ADX_Chaikin_OBOS_Strategy"
timeframe = "6h"
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

    # ADX calculation (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    for i in range(14, n):
        plus_sm = np.sum(plus_dm[i-13:i+1])
        minus_sm = np.sum(minus_dm[i-13:i+1])
        tr_sum = np.sum(tr[i-13:i+1])
        if tr_sum > 0:
            plus_di[i] = 100 * plus_sm / tr_sum
            minus_di[i] = 100 * minus_sm / tr_sum
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
        if i >= 27:  # ADX needs 14 + 14 periods
            adx[i] = np.mean(dx[i-13:i+1])

    # Chaikin Oscillator: (3-period EMA of ADL) - (10-period EMA of ADL)
    adl = np.zeros(n)
    for i in range(n):
        if high[i] == low[i]:
            adl[i] = 0
        else:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            adl[i] = adl[i-1] + clv * volume[i] if i > 0 else clv * volume[i]

    adl_ema3 = np.zeros(n)
    adl_ema10 = np.zeros(n)
    adl_ema3_smooth = np.zeros(n)
    adl_ema10_smooth = np.zeros(n)
    for i in range(n):
        if i == 0:
            adl_ema3[i] = adl[i]
            adl_ema10[i] = adl[i]
        else:
            adl_ema3[i] = 0.5 * adl[i] + 0.5 * adl_ema3[i-1]
            adl_ema10[i] = 2/11 * adl[i] + 9/11 * adl_ema10[i-1]
        adl_ema3_smooth[i] = adl_ema3[i]
        adl_ema10_smooth[i] = adl_ema10[i]
    chaikin = adl_ema3_smooth - adl_ema10_smooth

    # Get 1d close for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d
    ema_1d = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        else:
            ema_1d[i] = 2/51 * close_1d[i] + 49/51 * ema_1d[i-1]
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after ADX warmup
        if np.isnan(adx[i]) or np.isnan(chaikin[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Strong trend (ADX > 25): trend following
            if adx[i] > 25:
                # Uptrend: +DI > -DI and price above 1d EMA50
                if i < 14:  # DI not ready
                    continue
                plus_di_val = 100 * np.sum(plus_dm[i-13:i+1]) / np.sum(tr[i-13:i+1]) if np.sum(tr[i-13:i+1]) > 0 else 0
                minus_di_val = 100 * np.sum(minus_dm[i-13:i+1]) / np.sum(tr[i-13:i+1]) if np.sum(tr[i-13:i+1]) > 0 else 0
                if plus_di_val > minus_di_val and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Downtrend: -DI > +DI and price below 1d EMA50
                elif minus_di_val > plus_di_val and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Weak trend (ADX < 20): mean reversion at Chaikin extremes
            elif adx[i] < 20:
                # Oversold: Chaikin < -0.1 * mean absolute Chaikin (adaptive threshold)
                chaikin_ma = np.mean(np.abs(chaikin[max(0, i-20):i+1]))
                if chaikin[i] < -0.1 * chaikin_ma and close[i] < ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Overbought: Chaikin > 0.1 * mean absolute Chaikin
                elif chaikin[i] > 0.1 * chaikin_ma and close[i] > ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakening (ADX < 20) OR Chaikin becomes overbought
            chaikin_ma = np.mean(np.abs(chaikin[max(0, i-20):i+1]))
            if adx[i] < 20 or chaikin[i] > 0.1 * chaikin_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakening (ADX < 20) OR Chaikin becomes oversold
            chaikin_ma = np.mean(np.abs(chaikin[max(0, i-20):i+1]))
            if adx[i] < 20 or chaikin[i] < -0.1 * chaikin_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals