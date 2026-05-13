#!/usr/bin/env python3
# 6h_TripleBarrier_Breakout_12hTrend_Volume
# Hypothesis: Combines Donchian channel breakout with 12h trend filter and volume confirmation.
# Uses triple barrier: entry only when price breaks Donchian(20) high/low, closes outside 1.5*ATR band,
# and volume exceeds 2x 20-period average. Designed for low-frequency, high-conviction trades
# that work in both bull and bear markets by requiring strong momentum and volume confirmation.

name = "6h_TripleBarrier_Breakout_12hTrend_Volume"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h trend: EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility band (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Triple barrier conditions
        price_above_donchian = close[i] > donchian_high[i]
        price_below_donchian = close[i] < donchian_low[i]
        close_outside_upper = close[i] > (donchian_high[i] + 1.5 * atr[i])
        close_outside_lower = close[i] < (donchian_low[i] - 1.5 * atr[i])
        vol_spike = volume_spike[i]
        
        # Trend conditions
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]

        if position == 0:
            # LONG: Price breaks above Donchian high, closes outside upper band, uptrend, volume spike
            if price_above_donchian and close_outside_upper and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, closes outside lower band, downtrend, volume spike
            elif price_below_donchian and close_outside_lower and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian low OR trend reversal
            if close[i] < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian high OR trend reversal
            if close[i] > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals