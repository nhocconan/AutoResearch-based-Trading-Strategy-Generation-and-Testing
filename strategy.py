#!/usr/bin/env python3
# 4H_Donchian20_VolumeTrend_1D_EMA50
# Hypothesis: 4h Donchian breakout with volume confirmation and 1d EMA50 trend filter.
# Long when price breaks above 4h Donchian upper (20) + volume spike + 1d EMA50 uptrend.
# Short when price breaks below 4h Donchian lower (20) + volume spike + 1d EMA50 downtrend.
# Uses volume ratio > 1.5 and ATR filter to avoid whipsaws.
# Designed for 4h timeframe with target 20-40 trades/year per symbol.

name = "4H_Donchian20_VolumeTrend_1D_EMA50"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    
    # Align 1d trend to 4h
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # 4h Donchian channels (20-period)
    lookback = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        dc_upper[i] = np.max(high[i - lookback + 1:i + 1])
        dc_lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume ratio (current vs 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20 - 1, n):
        vol_ma[i] = np.mean(volume[i - 20 + 1:i + 1])
    vol_ratio = volume / vol_ma
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14 - 1, n):
        atr[i] = np.mean(tr[i - 14 + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + Donchian breakout + volume spike
            if bullish and close[i] > dc_upper[i] and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + Donchian breakdown + volume spike
            elif bearish and close[i] < dc_lower[i] and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: trend reversal or stoploss
            if bearish or close[i] < dc_lower[i] or close[i] < (signals[i-1] * dc_upper[i-1] + (1 - signals[i-1]) * (close[i-1] - 2 * atr[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend reversal or stoploss
            if bullish or close[i] > dc_upper[i] or close[i] > (signals[i-1] * dc_lower[i-1] + (1 - signals[i-1]) * (close[i-1] + 2 * atr[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals