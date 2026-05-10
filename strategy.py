#!/usr/bin/env python3
"""
6h_Engulfing_1dTrend_Volume
Hypothesis: Use daily trend (EMA34) and volume spikes to filter daily candlestick engulfing patterns on 6h timeframe.
Engulfing patterns signal strong momentum shifts; combining with daily trend and volume reduces false signals.
Designed for ~15-25 trades/year on 6h timeframe to avoid excessive fee drift.
"""

name = "6h_Engulfing_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 6h price and volume
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    # Bullish engulfing: current green candle fully engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price < close) & (close > open_price) & \
                     (open_price < close) & (close > open_price) & \
                     (open_price < close) & (close > open_price)
    bullish_engulf = (close > open_price) & (open_price < close) & (close > open_price) & \
                     (open_price < close) & (close > open_price) & \
                     (open_price < close) & (close > open_price)
    # Correct bullish engulfing: current green candle body completely engulfs previous red candle body
    bullish_engulf = (close > open_price) & \
                     (open_price < close) & \
                     (close[1:] > open_price[:-1]) & \
                     (open_price[1:] < close[:-1]) & \
                     (close[1:] - open_price[1:] > open_price[:-1] - close[:-1])
    # Fix indexing: shift to align current candle with previous
    bullish = (close > open_price) & \
              (np.roll(close, 1) < np.roll(open_price, 1)) & \
              (close > np.roll(open_price, 1)) & \
              (open_price < np.roll(close, 1))
    # Bearish engulfing: current red candle body completely engulfs previous green candle body
    bearish = (close < open_price) & \
              (np.roll(close, 1) > np.roll(open_price, 1)) & \
              (close < np.roll(open_price, 1)) & \
              (open_price > np.roll(close, 1))
    
    # Handle first element (no previous candle)
    bullish[0] = False
    bearish[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34 days) and volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema_34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing AND uptrend (above daily EMA34) AND volume spike
            if bullish[i] and close[i] > ema_34_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing AND downtrend (below daily EMA34) AND volume spike
            elif bearish[i] and close[i] < ema_34_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish engulfing OR trend turns bearish
            if bearish[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish engulfing OR trend turns bullish
            if bullish[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals