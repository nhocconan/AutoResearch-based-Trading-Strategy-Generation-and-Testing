#!/usr/bin/env python3
# 6h_Weekly_Pivot_Swing_Rejection
# Hypothesis: Trade reversals at weekly pivot levels (R1/S1) in the direction of 1d trend, confirmed by volume exhaustion and momentum divergence.
# In bull markets: buy R1 support when price shows bullish divergence on RSI and holds above 1d EMA50.
# In bear markets: sell S1 resistance when price shows bearish divergence and holds below 1d EMA50.
# Weekly pivots provide strong institutional support/resistance. Rejections at these levels with momentum divergence offer high-probability swings.
# Works in both bull (buy support dips) and bear (sell resistance rallies) via trend-aligned mean reversion.

name = "6h_Weekly_Pivot_Swing_Rejection"
timeframe = "6h"
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

    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot points (using prior week's OHLC)
    high_wk = df_1w['high'].values
    low_wk = df_1w['low'].values
    close_wk = df_1w['close'].values
    
    # Calculate pivot and levels
    pivot = (high_wk + low_wk + close_wk) / 3.0
    r1 = 2 * pivot - low_wk
    s1 = 2 * pivot - high_wk
    r2 = pivot + (high_wk - low_wk)
    s2 = pivot - (high_wk - low_wk)
    
    # Get daily data for trend and momentum
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI for momentum divergence (14-period)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume exhaustion: volume < 0.5 * 24-period average (1 day worth at 6h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_exhaustion = volume < 0.5 * vol_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at S1 support + bullish RSI divergence + above 1d EMA50 + volume exhaustion
            # Bullish divergence: price making lower low, RSI making higher low
            bullish_div = False
            if i >= 2:
                if low[i] < low[i-1] and low[i-1] < low[i-2] and rsi_aligned[i] > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2]:
                    bullish_div = True
            
            if (abs(low[i] - s1_aligned[i]) < 0.001 * s1_aligned[i] or low[i] <= s1_aligned[i]) and \
               bullish_div and \
               close[i] > ema50_1d_aligned[i] and \
               volume_exhaustion[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R1 resistance + bearish RSI divergence + below 1d EMA50 + volume exhaustion
            # Bearish divergence: price making higher high, RSI making lower high
            bearish_div = False
            if i >= 2:
                if high[i] > high[i-1] and high[i-1] > high[i-2] and rsi_aligned[i] < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2]:
                    bearish_div = True
            
            elif (abs(high[i] - r1_aligned[i]) < 0.001 * r1_aligned[i] or high[i] >= r1_aligned[i]) and \
                 bearish_div and \
                 close[i] < ema50_1d_aligned[i] and \
                 volume_exhaustion[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or RSI overbought or trend breaks
            if close[i] >= r1_aligned[i] or rsi_aligned[i] > 70 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or RSI oversold or trend breaks
            if close[i] <= s1_aligned[i] or rsi_aligned[i] < 30 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals