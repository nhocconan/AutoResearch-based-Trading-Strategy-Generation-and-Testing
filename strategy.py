#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # Camarilla pivot levels from previous day
    def calculate_camarilla(h, l, c):
        R4 = c + ((h - l) * 1.1 / 2)
        R3 = c + ((h - l) * 1.1 / 4)
        R2 = c + ((h - l) * 1.1 / 6)
        R1 = c + ((h - l) * 1.1 / 12)
        S1 = c - ((h - l) * 1.1 / 12)
        S2 = c - ((h - l) * 1.1 / 6)
        S3 = c - ((h - l) * 1.1 / 4)
        S4 = c - ((h - l) * 1.1 / 2)
        return R1, R2, R3, R4, S1, S2, S3, S4
    
    # Use previous day's OHLC for Camarilla levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    R1, R2, R3, R4, S1, S2, S3, S4 = calculate_camarilla(prev_high, prev_low, prev_close)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i - 20]
        if i < 19:
            vol_ma20[i] = np.nan
        else:
            vol_ma20[i] = vol_sum / 20
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~12 hours
    
    start_idx = 19  # Ensure volume MA is valid
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above R3 AND 1d uptrend AND volume filter
            if close[i] > R3[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S3 AND 1d downtrend AND volume filter
            elif close[i] < S3[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below R1 OR trend turns down
            if close[i] < R1[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above S1 OR trend turns up
            if close[i] > S1[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Captures institutional breakout patterns in both bull and bear markets.
# Long when price breaks above R3 in 1d uptrend with volume surge,
# short when breaks below S3 in 1d downtrend with volume surge.
# Uses tight entry conditions to limit trades and reduce fee drag.
# Position size 0.25 manages risk, cooldown of 3 bars limits frequency.