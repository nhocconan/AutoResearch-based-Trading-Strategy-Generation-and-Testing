#!/usr/bin/env python3
# 12h_Keltner_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Price breaks out of Keltner Channel with low volatility squeeze, confirmed by 1d trend and volume spike.
# Long: Close > upper Keltner + 1d uptrend + volume spike
# Short: Close < lower Keltner + 1d downtrend + volume spike
# Exit: Opposite Keltner band or trend reversal
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Target: 15-40 trades/year to minimize fee drag.

name = "12h_Keltner_Squeeze_Breakout_1dTrend_Volume"
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

    # Get 1d data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend (50-period)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for Keltner Channel (20-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 20-period EMA for Keltner basis
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channels: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # Align 1d indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.5 * 5-period average
    vol_ma_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    volume_spike = volume > 2.5 * vol_ma_5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper Keltner + 1d uptrend + volume spike
            if close[i] > keltner_upper[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower Keltner + 1d downtrend + volume spike
            elif close[i] < keltner_lower[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below lower Keltner or trend reversal
            if close[i] < keltner_lower[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above upper Keltner or trend reversal
            if close[i] > keltner_upper[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals