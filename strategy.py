#!/usr/bin/env python3
# 6h_Engulfing_1dTrend_VolumeSpike
# Hypothesis: 6h bullish/bearish engulfing candles with daily EMA50 trend filter and volume spike confirmation.
# Works in bull/bear by trading reversal patterns aligned with higher timeframe trend.
# Targets 15-30 trades/year to minimize fee drag.

name = "6h_Engulfing_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 2.0
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                         (close[i] > open_price[i-1]) and (open_price[i] < close[i-1])
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                         (close[i] < open_price[i-1]) and (open_price[i] > close[i-1])
        
        if position == 0:
            # Long: bullish engulfing with uptrend and volume spike
            if bullish_engulf and trend_1d_up_aligned[i] > 0.5 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing with downtrend and volume spike
            elif bearish_engulf and trend_1d_down_aligned[i] > 0.5 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish engulfing or trend fails
            if bearish_engulf or trend_1d_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish engulfing or trend fails
            if bullish_engulf or trend_1d_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals