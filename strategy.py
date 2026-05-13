#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Bounce_1dTrend_Reversal
# Hypothesis: Enter long at Camarilla S1 support during 1d uptrend when price shows rejection (close > open) and volume confirms.
# Enter short at Camarilla R1 resistance during 1d downtrend when price shows rejection (close < open) and volume confirms.
# Uses Camarilla levels from daily timeframe for institutional support/resistance.
# Trend filter ensures alignment with higher timeframe momentum.
# Works in bull (bounces at S1 in uptrend) and bear (rejections at R1 in downtrend).
# Low frequency due to specific price action requirements at pivot levels.

name = "12h_Camarilla_Pivot_Bounce_1dTrend_Reversal"
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
    open_price = prices['open'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Using previous day's high, low, close to avoid look-ahead
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla levels: H5, L5, H4, L4, H3, L3, H2, L2, H1, L1
    # We focus on H3 (resistance) and L3 (support) for reversals
    range_val = phigh - plow
    h3 = pclose + range_val * 1.1 / 4
    l3 = pclose - range_val * 1.1 / 4
    h4 = pclose + range_val * 1.1 / 2
    l4 = pclose - range_val * 1.1 / 2
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Price action confirmation: rejection candle
    # Bullish rejection: close > open (bullish candle)
    # Bearish rejection: close < open (bearish candle)
    bullish_rejection = close > open_price
    bearish_rejection = close < open_price
    
    # Volume confirmation: volume > 1.5 * 4-period average (1 day worth at 12h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_confirm = volume > 1.5 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at L3 support, bullish rejection, uptrend, volume
            if (low[i] <= l3_aligned[i] * 1.005 and  # Allow small tolerance for wick
                close[i] > l3_aligned[i] and
                bullish_rejection[i] and
                close[i] > ema50_1d_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at H3 resistance, bearish rejection, downtrend, volume
            elif (high[i] >= h3_aligned[i] * 0.995 and  # Allow small tolerance for wick
                  close[i] < h3_aligned[i] and
                  bearish_rejection[i] and
                  close[i] < ema50_1d_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches L4 (stop) or H3 (target) or trend reversal
            if close[i] <= l4_aligned[i] or close[i] >= h3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches H4 (stop) or L3 (target) or trend reversal
            if close[i] >= h4_aligned[i] or close[i] <= l3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals