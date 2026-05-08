#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily volatility breakout with volume confirmation and trend filter.
# Combines daily ATR breakout (buy when price breaks above close + ATR, sell when breaks below close - ATR)
# with volume spike confirmation and 12h EMA(50) trend filter. Uses daily timeframe for volatility
# measurement to avoid noise, 12h for execution to reduce frequency. Designed for low trade frequency
# (15-25/year) to minimize fee drag while capturing volatility expansion moves in both bull and bear markets.

name = "12h_DailyATR_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily timeframe
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.zeros(len(close_1d))
    atr_14[14] = np.mean(tr[:14])
    for i in range(15, len(close_1d)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i-1]) / 14
    
    # Calculate upper and lower bands: close ± ATR
    upper_band = close_1d + atr_14
    lower_band = close_1d - atr_14
    
    # Align bands to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # 12h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 12h volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper band with volume confirmation and uptrend
            if close[i] > upper_aligned[i] and vol_confirm[i] and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band with volume confirmation and downtrend
            elif close[i] < lower_aligned[i] and vol_confirm[i] and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA50 or reverses below lower band
            if close[i] < ema_50[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA50 or reverses above upper band
            if close[i] > ema_50[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals