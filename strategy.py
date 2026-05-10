#!/usr/bin/env python3
# 6h_Engulfing_Pattern_1dTrend_Volume
# Hypothesis: 6-hour bullish/bearish engulfing candlestick patterns with daily trend filter (EMA50) and volume confirmation.
# Engulfing patterns signal strong momentum reversals; daily EMA50 filters trend direction to avoid counter-trend trades;
# Volume confirmation ensures pattern reliability. Designed for 6h to achieve 12-37 trades/year, suitable for both bull and bear markets.

name = "6h_Engulfing_Pattern_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 6h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish engulfing: current green candle completely engulfs previous red candle
        bullish_engulf = (close[i] > open_[i]) and (open_[i] < close[i-1]) and (close[i] > open_[i-1]) and (open_[i-1] > close[i-1])
        # Bearish engulfing: current red candle completely engulfs previous green candle
        bearish_engulf = (close[i] < open_[i]) and (open_[i] > close[i-1]) and (close[i] < open_[i-1]) and (open_[i-1] < close[i-1])
        
        if position == 0:
            # Long: bullish engulfing, above daily EMA50, strong volume
            if bullish_engulf and close[i] > ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing, below daily EMA50, strong volume
            elif bearish_engulf and close[i] < ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish engulfing or price drops below daily EMA50
            if bearish_engulf or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish engulfing or price rises above daily EMA50
            if bullish_engulf or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals