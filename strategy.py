#!/usr/bin/env python3
"""
4h_1d_ExponentialMovingAverage34_Slope_Direction_Trend_Follower
Hypothesis: EMA34 slope on daily timeframe captures intermediate trend direction.
Trades in direction of daily EMA34 slope using 4h price action with volume confirmation.
Designed to work in both bull and bear markets by following the trend on higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for EMA34 slope
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    
    # Calculate EMA34 on daily close
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate EMA34 slope (change over 3 days)
    ema34_slope_daily = np.zeros_like(ema34_daily)
    ema34_slope_daily[3:] = (ema34_daily[3:] - ema34_daily[:-3]) / 3
    
    # Align daily EMA34 slope to 4h timeframe
    ema34_slope_aligned = align_htf_to_ltf(prices, df_daily, ema34_slope_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h ATR for stop loss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_slope_aligned[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        slope = ema34_slope_aligned[i]
        atr = atr_4h[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: positive EMA34 slope (uptrend) with volume confirmation
            if slope > 0 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: negative EMA34 slope (downtrend) with volume confirmation
            elif slope < 0 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA34 slope turns negative or ATR-based stop
            if slope < 0 or (i > 0 and close[i-1] > price + 0.5 * atr and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA34 slope turns positive or ATR-based stop
            if slope > 0 or (i > 0 and close[i-1] < price - 0.5 * atr and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_EMA34_Slope_Direction_Trend_Follower"
timeframe = "4h"
leverage = 1.0