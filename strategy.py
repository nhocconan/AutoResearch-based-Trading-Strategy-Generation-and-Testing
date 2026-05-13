#!/usr/bin/env python3
# 6h_Liquidity_Imbalance_Reversal_1dTrend_Volume
# Hypothesis: Detect liquidity imbalances at 6h swing highs/lows that precede reversals.
# A liquidity imbalance occurs when price rapidly moves through a level with little opposing volume,
# leaving unfilled orders. We identify these as 6h swing points where the subsequent 3-bar move
# exceeds ATR(14) with below-average volume on the move. Reversal entries occur when price
# returns to test these imbalances with volume confirmation, filtered by 1d EMA50 trend.
# Works in both bull/bear markets by fading imbalance-driven moves in the direction of the
# higher timeframe trend. Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Liquidity_Imbalance_Reversal_1dTrend_Volume"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate ATR(14) for 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Volume average for imbalance detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Identify liquidity imbalances: swing points with strong subsequent move on low volume
    bullish_imb = np.zeros(n, dtype=bool)  # liquidity vacuum below (for long entries)
    bearish_imb = np.zeros(n, dtype=bool)  # liquidity vacuum above (for short entries)

    for i in range(2, n - 3):
        # Bullish imbalance: swing low followed by strong up move on weak volume
        if low[i] <= low[i-1] and low[i] <= low[i+1]:  # swing low
            move_up = high[i+2] - low[i]  # 3-bar high minus swing low
            vol_move = (volume[i+1] + volume[i+2]) / 2  # avg volume on move
            if move_up > atr[i] * 1.5 and vol_move < vol_avg_20[i] * 0.8:
                bullish_imb[i] = True  # liquidity vacuum below swing low
        
        # Bearish imbalance: swing high followed by strong down move on weak volume
        if high[i] >= high[i-1] and high[i] >= high[i+1]:  # swing high
            move_down = high[i] - low[i+2]  # swing high minus 3-bar low
            vol_move = (volume[i+1] + volume[i+2]) / 2  # avg volume on move
            if move_down > atr[i] * 1.5 and vol_move < vol_avg_20[i] * 0.8:
                bearish_imb[i] = True  # liquidity vacuum above swing high

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price tests bullish imbalance (liquidity vacuum below) with volume confirmation
            # in the direction of 1d trend (above EMA50 = bullish)
            if bullish_imb[i] and close[i] > ema_50_1d_aligned[i]:
                # Look for retest of swing low with volume
                if low[i] <= close[i] and volume[i] > vol_avg_20[i] * 1.2:
                    signals[i] = 0.25
                    position = 1
            # SHORT: price tests bearish imbalance (liquidity vacuum above) with volume confirmation
            # in the direction of 1d trend (below EMA50 = bearish)
            elif bearish_imb[i] and close[i] < ema_50_1d_aligned[i]:
                # Look for retest of swing high with volume
                if high[i] >= close[i] and volume[i] > vol_avg_20[i] * 1.2:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks above recent high or violates trend
            if high[i] > np.max(high[max(0, i-5):i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks below recent low or violates trend
            if low[i] < np.min(low[max(0, i-5):i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals