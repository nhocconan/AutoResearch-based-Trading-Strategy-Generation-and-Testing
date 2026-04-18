#!/usr/bin/env python3
"""
12h_HeikinAshi_Trend_Filter_V1
12h strategy using Heikin Ashi candles for trend detection with volume confirmation and momentum filter.
- Long: HA close > HA open + volume > 1.2x average + RSI(14) > 50
- Short: HA close < HA open + volume > 1.2x average + RSI(14) < 50
- Exit: Opposite HA candle color or RSI crossing 50
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (sustained uptrends) and bear markets (sustained downtrends)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Heikin Ashi candles
    ha_close = (high + low + close + open_price) / 4
    ha_open = np.zeros_like(close)
    ha_open[0] = (open_price[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    
    # Get daily data for volume average and RSI
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # HA candle color
        ha_bullish = ha_close[i] > ha_open[i]
        ha_bearish = ha_close[i] < ha_open[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.2 * vol_ma_aligned[i]
        
        # Momentum filter
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        if position == 0:
            # Long: bullish HA + volume + RSI > 50
            if ha_bullish and vol_confirm and rsi_bullish:
                signals[i] = 0.25
                position = 1
            # Short: bearish HA + volume + RSI < 50
            elif ha_bearish and vol_confirm and rsi_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish HA or RSI < 50
            if ha_bearish or not rsi_bullish:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish HA or RSI > 50
            if ha_bullish or rsi_bullish:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HeikinAshi_Trend_Filter_V1"
timeframe = "12h"
leverage = 1.0