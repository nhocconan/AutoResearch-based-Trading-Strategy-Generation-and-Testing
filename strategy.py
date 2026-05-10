#!/usr/bin/env python3
# 4h_Engulfing_Trend_Trig
# Hypothesis: Bullish/bearish engulfing candles on 4h aligned with daily trend (EMA34) and volume > 1.5x 20-bar MA capture momentum moves. Engulfing patterns signal strong reversals/continuations; trend filter avoids counter-trend trades; volume filter ensures conviction. Works in bull markets by buying dips in uptrend and in bear markets by selling rallies in downtrend. Target: 20-40 trades/year.

name = "4h_Engulfing_Trend_Trig"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Engulfing detection
    bullish_engulf = (close > open_) & (open_ < close) & (close >= open_) & (open_ <= close) & \
                     (close >= open_.shift(1)) & (open_ <= close.shift(1)) & \
                     (close > close.shift(1)) & (open_ < open_.shift(1))
    bearish_engulf = (open_ > close) & (close < open_) & (open_ >= close) & (close <= open_) & \
                     (open_ >= close.shift(1)) & (close <= open_.shift(1)) & \
                     (open_ > open_.shift(1)) & (close < close.shift(1))
    # Fix: proper engulfing conditions
    bullish_engulf = (close > open_) & (open_.shift(1) > close.shift(1)) & (close >= open_.shift(1)) & (open_ <= close.shift(1))
    bearish_engulf = (open_ > close) & (close.shift(1) > open_.shift(1)) & (open_ >= close.shift(1)) & (close <= open_.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (>1.5x MA)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: bullish engulf + uptrend + volume
            if bullish_engulf[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish engulf + downtrend + volume
            elif bearish_engulf[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or bearish engulf
            if not uptrend or bearish_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or bullish engulf
            if not downtrend or bullish_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals