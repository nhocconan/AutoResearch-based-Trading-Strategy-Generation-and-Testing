#!/usr/bin/env python3
# 12h_DeMark_TD_Setup_1dTrend_Volume
# Hypothesis: Sequential (TD Setup) 9-count on 12h indicates exhaustion; trade opposite direction on 12h close with 1d trend filter and volume confirmation.
# Works in bull/bear by following 1d trend direction; TD Setup identifies exhaustion points; volume confirms institutional participation.
# Target: 15-30 trades/year per symbol to minimize fee drag.

name = "12h_DeMark_TD_Setup_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate TD Setup (Sequential) on 12h close
    # Buy Setup: close < low 4 bars ago (for bears)
    # Sell Setup: close > high 4 bars ago (for bulls)
    # We count consecutive setups up to 9
    buy_setup = close < np.roll(low, 4)
    sell_setup = close > np.roll(high, 4)
    
    # Initialize count arrays
    buy_count = np.zeros(n, dtype=int)
    sell_count = np.zeros(n, dtype=int)
    
    # Count consecutive setups
    for i in range(4, n):
        if buy_setup[i]:
            buy_count[i] = buy_count[i-1] + 1 if i > 0 else 1
            sell_count[i] = 0  # reset opposite count
        elif sell_setup[i]:
            sell_count[i] = sell_count[i-1] + 1 if i > 0 else 1
            buy_count[i] = 0
        else:
            buy_count[i] = 0
            sell_count[i] = 0

    # TD Setup signals: 9-count exhaustion
    td_buy_setup = (buy_count == 9)  # Bearish exhaustion - potential long
    td_sell_setup = (sell_count == 9)  # Bullish exhaustion - potential short

    # Trend filter: 1d EMA34
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: current volume > 1.8 x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(35, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TD Sell Setup 9 (bullish exhaustion) in uptrend with volume
            if (td_sell_setup[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TD Buy Setup 9 (bearish exhaustion) in downtrend with volume
            elif (td_buy_setup[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TD Buy Setup 9 or trend turns down
            if td_buy_setup[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TD Sell Setup 9 or trend turns up
            if td_sell_setup[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals