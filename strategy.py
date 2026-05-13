#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from daily chart identify key support/resistance.
# Break above R1 with 1d uptrend and volume confirmation = long.
# Break below S1 with 1d downtrend and volume confirmation = short.
# Exit on opposite pivot level or trend reversal. Designed for low-frequency, high-conviction trades.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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

    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels: R1, S1
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Pivot + Range * 1.1 / 12
    # S1 = Pivot - Range * 1.1 / 12
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    pivot = (H + L + C) / 3.0
    range_hl = H - L
    R1 = pivot + range_hl * 1.1 / 12.0
    S1 = pivot - range_hl * 1.1 / 12.0
    
    # Align pivot levels to 4h timeframe (no extra delay needed for pivot levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 2.0 * 20-period average (high threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # Trend conditions
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above R1 + uptrend + volume spike
            if price_above_R1 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + downtrend + volume spike
            elif price_below_S1 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR trend reversal
            if price_below_S1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR trend reversal
            if price_above_R1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals