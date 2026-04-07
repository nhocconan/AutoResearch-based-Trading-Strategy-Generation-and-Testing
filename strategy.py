#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-day EMA200 trend filter
# Long when Alligator jaws (13-period SMMA) above teeth (8-period SMMA) above lips (5-period SMMA),
# price above 1-day EMA200, and close > lips
# Short when jaws below teeth below lips (alligator sleeping),
# price below 1-day EMA200, and close < lips
# Exit when Alligator lines cross (jaws crosses teeth) or price crosses EMA200
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Williams Alligator is a trend-following indicator that works well in trending markets
# and avoids whipsaws in ranging markets by requiring alignment of three smoothed lines
# Designed to work in both bull and bear markets by requiring EMA200 trend alignment

name = "12h_williams_alligator_1d_ema200_v1"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Smoothing"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    # First value is simple moving average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator lines cross (jaw crosses teeth) or price crosses below EMA200
            elif jaw_aligned[i] < teeth_aligned[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator lines cross (jaw crosses teeth) or price crosses above EMA200
            elif jaw_aligned[i] > teeth_aligned[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Alligator alignment and EMA200 trend
            # Long: jaws > teeth > lips (alligator awake, uptrend), price above EMA200, close > lips
            if (jaw_aligned[i] > teeth_aligned[i] and
                teeth_aligned[i] > lips_aligned[i] and
                close[i] > ema_200_aligned[i] and
                close[i] > lips_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: jaws < teeth < lips (alligator awake, downtrend), price below EMA200, close < lips
            elif (jaw_aligned[i] < teeth_aligned[i] and
                  teeth_aligned[i] < lips_aligned[i] and
                  close[i] < ema_200_aligned[i] and
                  close[i] < lips_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals