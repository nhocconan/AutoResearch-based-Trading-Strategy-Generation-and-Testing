#!/usr/bin/env python3
# 1d_Weekly_Keltner_Channel_Breakout_Trend_Volume
# Hypothesis: Breakout above weekly Keltner upper band or below lower band in the direction of daily trend, confirmed by volume spike.
# Weekly Keltner channels (ATR-based) provide dynamic support/resistance; breakouts indicate momentum with volatility adjustment.
# Daily trend filter (EMA50) ensures alignment with higher timeframe momentum. Volume spike confirms institutional participation.
# Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend).
# Low frequency due to weekly volatility-based bands and strict volume confirmation.

name = "1d_Weekly_Keltner_Channel_Breakout_Trend_Volume"
timeframe = "1d"
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

    # Get weekly data for Keltner channel calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 and ATR(10)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr_1w = np.maximum(np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1])), np.abs(low_1w[1:] - close_1w[:-1]))
    tr_1w = np.concatenate([[np.nan], tr_1w])  # align with original index
    atr10_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly Keltner channels: EMA20 ± 2 * ATR(10)
    keltner_upper = ema20_1w + 2 * atr10_1w
    keltner_lower = ema20_1w - 2 * atr10_1w
    
    # Align weekly Keltner channels to daily timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 20-period average (approx 20 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly Keltner upper + daily uptrend + volume spike
            if close[i] > keltner_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly Keltner lower + daily downtrend + volume spike
            elif close[i] < keltner_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly EMA20 (middle of Keltner) OR trend reversal
            keltner_middle_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
            if close[i] < keltner_middle_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly EMA20 OR trend reversal
            keltner_middle_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
            if close[i] > keltner_middle_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals