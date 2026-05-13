#!/usr/bin/env python3
# 12h_1d_KC_Breakout_Trend_Volume
# Hypothesis: Keltner Channel (ATR-based) breakouts on 12h chart capture strong moves.
# Long: price breaks above upper KC (EMA20 + 2*ATR) with 1d uptrend and volume spike.
# Short: price breaks below lower KC (EMA20 - 2*ATR) with 1d downtrend and volume spike.
# Exit on opposite KC touch or trend reversal. Designed for low-frequency, high-conviction trades in both bull and bear markets.

name = "12h_1d_KC_Breakout_Trend_Volume"
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
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Keltner Channel on 12h: EMA20 +/- 2*ATR(10)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema20 + 2 * atr10
    kc_lower = ema20 - 2 * atr10
    
    # Volume spike: volume > 2.0 * 20-period average (high threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        price_above_kc_upper = close[i] > kc_upper[i]
        price_below_kc_lower = close[i] < kc_lower[i]
        
        # Trend conditions
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above upper KC + uptrend + volume spike
            if price_above_kc_upper and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower KC + downtrend + volume spike
            elif price_below_kc_lower and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches lower KC OR trend reversal
            if close[i] < kc_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches upper KC OR trend reversal
            if close[i] > kc_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals