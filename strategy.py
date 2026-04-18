#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R (14) extreme reversals with 12h EMA(34) trend filter and volume confirmation.
In oversold conditions (WR < -80) with bullish 12h trend and volume spike → long.
In overbought conditions (WR > -20) with bearish 12h trend and volume spike → short.
Weekly volatility filter avoids choppy markets. Designed for 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R."""
    if len(high) < period:
        return np.full(len(close), np.nan)
    
    highest_high = np.full(len(high), np.nan)
    lowest_low = np.full(len(low), np.nan)
    
    for i in range(period-1, len(high)):
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    
    wr = np.full(len(close), np.nan)
    for i in range(period-1, len(close)):
        if highest_high[i] != lowest_low[i]:
            wr[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            wr[i] = -50
    
    return wr

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * 2 / (period + 1)) + ema[i-1] * (1 - 2 / (period + 1))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(34) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Get 1w data for volatility filter (ATR-based)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 12h
    ema_34_12h = calculate_ema(close_12h, 34)
    
    # Calculate ATR(14) on 1w for volatility filter
    tr1 = np.zeros(len(high_1w))
    tr2 = np.zeros(len(high_1w))
    tr3 = np.zeros(len(high_1w))
    tr1[1:] = np.abs(high_1w[1:] - low_1w[:-1])
    tr2[1:] = np.abs(high_1w[1:] - close_1w[:-1])
    tr3[1:] = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = np.zeros(len(tr))
    for i in range(14, len(tr)):
        atr_14_1w[i] = np.mean(tr[i-14:i])
    atr_ma_1w = np.zeros(len(tr))
    for i in range(28, len(tr)):  # 2-period MA of ATR
        atr_ma_1w[i] = np.mean(atr_14_1w[i-2:i])
    
    # Align to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    atr_14_1w_4h = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    atr_ma_1w_4h = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    # Calculate Williams %R on 4h
    wr_14 = calculate_williams_r(high, low, close, 14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # need Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_14[i]) or np.isnan(ema_34_12h_4h[i]) or 
            np.isnan(atr_14_1w_4h[i]) or np.isnan(atr_ma_1w_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_filter = atr_14_1w_4h[i] > 0.5 * atr_ma_1w_4h[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80), bullish 12h trend, volume confirmation, not low vol
            if wr_14[i] < -80 and close[i] > ema_34_12h_4h[i] and vol_confirmed and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), bearish 12h trend, volume confirmation, not low vol
            elif wr_14[i] > -20 and close[i] < ema_34_12h_4h[i] and vol_confirmed and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50 or trend turns bearish
            if wr_14[i] > -50 or close[i] <= ema_34_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50 or trend turns bullish
            if wr_14[i] < -50 or close[i] >= ema_34_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_12hEMA34_Volume_VolFilter"
timeframe = "4h"
leverage = 1.0