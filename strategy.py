#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and ATR volatility filter
# Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA50 AND ATR(14) < 1.5 * ATR(50) (low volatility regime)
# Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA50 AND ATR(14) < 1.5 * ATR(50)
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R identifies extreme momentum exhaustion points that often precede reversals
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
# ATR volatility filter ensures entries occur during stable volatility regimes, reducing whipsaw
# Works in both bull and bear markets by fading extremes in the direction of the 1d trend

name = "4h_WilliamsR_MeanRev_1dEMA50_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 4h Williams %R(14), ATR(14), ATR(50) ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    
    # Calculate ATR(14) and ATR(50)
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_50 = tr.rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed bars)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_4h, atr_50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ATR volatility filter: ATR(14) < 1.5 * ATR(50) (low volatility regime)
    vol_filter = atr_14_aligned < (1.5 * atr_50_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: oversold AND uptrend AND low volatility regime
            if williams_r_aligned[i] < -80 and close[i] > ema50_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: overbought AND downtrend AND low volatility regime
            elif williams_r_aligned[i] > -20 and close[i] < ema50_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum returning)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum returning)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals