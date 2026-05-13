#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels from 1d chart identify key support/resistance.
# Break above R3 or below S3 with 1w uptrend/downtrend and volume confirmation = entry.
# Exit on opposite Camarilla level (S1/R1) or trend reversal.
# Designed for low-frequency, high-conviction trades on 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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

    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    hl_range = df_1d['high'] - df_1d['low']
    close_1d = df_1d['close']
    
    R3 = close_1d + 1.1 * hl_range
    S3 = close_1d - 1.1 * hl_range
    R1 = close_1d + 0.275 * hl_range
    S1 = close_1d - 0.275 * hl_range
    
    # Camarilla levels require 1-bar confirmation after the daily bar closes
    R3_1d = align_htf_to_ltf(prices, df_1d, R3.values, additional_delay_bars=1)
    S3_1d = align_htf_to_ltf(prices, df_1d, S3.values, additional_delay_bars=1)
    R1_1d = align_htf_to_ltf(prices, df_1d, R1.values, additional_delay_bars=1)
    S1_1d = align_htf_to_ltf(prices, df_1d, S1.values, additional_delay_bars=1)
    
    # 1w trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: volume > 2.0 * 20-period average (high threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or 
            np.isnan(R1_1d[i]) or np.isnan(S1_1d[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        price_above_R3 = close[i] > R3_1d[i]
        price_below_S3 = close[i] < S3_1d[i]
        
        # Reversal conditions (exit at opposite levels)
        price_below_R1 = close[i] < R1_1d[i]
        price_above_S1 = close[i] > S1_1d[i]
        
        # Trend conditions
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above R3 + uptrend + volume spike
            if price_above_R3 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + downtrend + volume spike
            elif price_below_S3 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below R1 OR trend reversal
            if price_below_R1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S1 OR trend reversal
            if price_above_S1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals