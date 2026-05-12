# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_OrderBlock_Liquidity_Grab_Reversal
Hypothesis: Institutional order blocks and liquidity grabs create high-probability reversal zones.
On 6B timeframe, price often reverses after sweeping liquidity (equal highs/lows) and showing rejection from order blocks.
Combines: 1) Liquidity sweep detection (equal highs/lows + close reversal), 2) Order block identification (last opposite color candle before strong move), 3) 1-week trend filter for direction bias.
Works in both bull/bear markets by trading reversals at key institutional levels rather than directional trends.
"""

name = "6h_OrderBlock_Liquidity_Grab_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1-week data for trend filter (institutional bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA21 for trend bias
    weekly_close = df_1w['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)

    # Identify order blocks: last opposite color candle before strong momentum candle
    bullish_ob = np.zeros(n, dtype=bool)  # Potential long entry zones
    bearish_ob = np.zeros(n, dtype=bool)  # Potential short entry zones
    
    for i in range(2, n):
        # Bullish OB: last red candle before a strong green candle (close > open + 60% of range)
        if (close[i-1] < open_prices[i-1] and  # Red candle
            close[i] > open_prices[i] and      # Green candle
            (close[i] - open_prices[i]) > 0.6 * (high[i] - low[i])):  # Strong bullish candle
            bullish_ob[i-1] = True  # Mark the red candle as bullish OB
            
        # Bearish OB: last green candle before a strong red candle
        if (close[i-1] > open_prices[i-1] and  # Green candle
            close[i] < open_prices[i] and      # Red candle
            (open_prices[i] - close[i]) > 0.6 * (high[i] - low[i])):  # Strong bearish candle
            bearish_ob[i-1] = True  # Mark the green candle as bearish OB

    # Detect liquidity sweeps: equal highs/lows followed by reversal
    # Equal highs: current high within 0.1% of previous significant high
    equal_high = np.zeros(n, dtype=bool)
    equal_low = np.zeros(n, dtype=bool)
    
    for i in range(10, n):  # Need lookback for swing points
        # Find recent swing high (simple: highest high in last 20 periods)
        lookback_start = max(0, i-20)
        swing_high = np.max(high[lookback_start:i])
        swing_low = np.min(low[lookback_start:i])
        
        # Equal high: current high near swing high
        if high[i] >= 0.999 * swing_high:
            equal_high[i] = True
            
        # Equal low: current low near swing low
        if low[i] <= 1.001 * swing_low:
            equal_low[i] = True

    # Liquidity grab signals: sweep + immediate reversal
    bullish_setup = np.zeros(n, dtype=bool)  # Long setup: swept lows + bullish rejection
    bearish_setup = np.zeros(n, dtype=bool)  # Short setup: swept highs + bearish rejection
    
    for i in range(5, n):
        # Bullish: swept equal lows then closed above OB or showed strength
        if (equal_low[i] and 
            i >= 2 and
            close[i] > low[i] + 0.5 * (high[i] - low[i]) and  # Closed in upper 50% of range
            bullish_ob[i-2:i].any()):  # OB nearby
            bullish_setup[i] = True
            
        # Bearish: swept equal highs then closed below OB or showed weakness
        if (equal_high[i] and 
            i >= 2 and
            close[i] < high[i] - 0.5 * (high[i] - low[i]) and  # Closed in lower 50% of range
            bearish_ob[i-2:i].any()):  # OB nearby
            bearish_setup[i] = True

    # Volume confirmation: above average volume
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    high_volume = volume > 1.5 * volume_ma20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(25, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema21_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish setup + weekly uptrend bias OR counter-trend in ranging markets
            weekly_uptrend = close[i] > weekly_ema21_aligned[i]
            if (bullish_setup[i] and high_volume[i] and 
                (weekly_uptrend or abs(close[i] - weekly_ema21_aligned[i]) < 0.02 * weekly_ema21_aligned[i])):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish setup + weekly downtrend bias OR counter-trend in ranging markets
            elif (bearish_setup[i] and high_volume[i] and 
                  (not weekly_uptrend or abs(close[i] - weekly_ema21_aligned[i]) < 0.02 * weekly_ema21_aligned[i])):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish setup or price returns to weekly EMA
            if bearish_setup[i] and high_volume[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < weekly_ema21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish setup or price returns to weekly EMA
            if bullish_setup[i] and high_volume[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > weekly_ema21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals