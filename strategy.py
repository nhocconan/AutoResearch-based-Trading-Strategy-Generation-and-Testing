#!/usr/bin/env python3
# 1d_KeltnerChannel_Breakout_VolumeSpike_1wTrend
# Hypothesis: Breakouts from 1D Keltner Channel (EMA20 +/- 2*ATR10) confirmed by volume spike and 1W EMA50 trend.
# Keltner channels adapt to volatility, capturing breakouts in both trending and ranging markets.
# Volume surge confirms institutional participation; weekly trend filter avoids counter-trend trades.
# Designed for 15-25 trades/year on 1D timeframe to minimize fee drag.
# Works in bull/bear: long when price breaks above upper KC with volume and above weekly EMA;
# short when breaks below lower KC with volume and below weekly EMA.

name = "1d_KeltnerChannel_Breakout_VolumeSpike_1wTrend"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate EMA20 and ATR10 for Keltner Channel (daily)
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel bands
    upper_kc = ema20 + 2 * atr10
    lower_kc = ema20 - 2 * atr10

    # Weekly EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: current volume > 2.5 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(upper_kc[i]) or 
            np.isnan(lower_kc[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper KC with volume spike and above weekly EMA50
            if (close[i] > upper_kc[i] and 
                volume_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower KC with volume spike and below weekly EMA50
            elif (close[i] < lower_kc[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA20 or weekly EMA50 turns down
            if close[i] < ema20[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA20 or weekly EMA50 turns up
            if close[i] > ema20[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals