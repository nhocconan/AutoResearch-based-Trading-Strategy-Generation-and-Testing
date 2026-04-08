#!/usr/bin/env python3
# 1d_1w_atr_breakout_v1
# Hypothesis: Daily ATR breakout with weekly trend filter. Long when price breaks above daily ATR-based upper band and weekly close > weekly open (bullish weekly candle). Short when price breaks below daily ATR-based lower band and weekly close < weekly open (bearish weekly candle). Exit when price re-enters the ATR band. Designed to capture trending moves with volatility-based entries and weekly trend confirmation to avoid counter-trend trades. Targets 20-50 trades per year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_atr_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ATR(14) for volatility-based bands
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR is just high-low
    atr = np.zeros(n)
    atr[:atr_period-1] = np.nan
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # ATR multiplier for band width
    atr_mult = 2.0
    
    # Calculate upper and lower bands (ATR-based channels)
    upper_band = close + atr_mult * atr
    lower_band = close - atr_mult * atr
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = df_1w['close'].values > df_1w['open'].values
    weekly_bearish = df_1w['close'].values < df_1w['open'].values
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after ATR warmup
        # Skip if data not ready
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price re-enters the ATR band (price <= upper_band and price >= lower_band)
            if price <= upper_band[i] and price >= lower_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price re-enters the ATR band
            if price <= upper_band[i] and price >= lower_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: ATR breakout with weekly trend alignment
            # Bullish: price breaks above upper band and weekly bullish
            if price > upper_band[i] and weekly_bullish_aligned[i] == 1.0:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below lower band and weekly bearish
            elif price < lower_band[i] and weekly_bearish_aligned[i] == 1.0:
                position = -1
                signals[i] = -0.25
    
    return signals