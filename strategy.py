#!/usr/bin/env python3
"""
4h_12h_TrueRange_Breakout_EMA_Trend_Volume
Hypothesis: Use 12h EMA for trend direction (filter whipsaws), 4h true range breakout above/below ATR-based channel for entry, and volume confirmation. True range captures volatility expansion while EMA trend filter ensures alignment with higher timeframe momentum. Targets 20-30 trades/year by requiring EMA trend alignment, volatility breakout beyond 1.5x ATR, and volume > 1.3x 20-period average. Works in bull markets by taking long breaks above upper band when 12h EMA rising, and in bear markets by taking short breaks below lower band when 12h EMA falling. Volatility breakout adapts to changing market conditions, reducing false signals in low volatility periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34)
    ema_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema_12h[33] = np.mean(close_12h[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align EMA to 4h timeframe (wait for bar close)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate ATR(14) for volatility bands
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate volatility bands: ±1.5 * ATR from close
    upper_band = close + 1.5 * atr
    lower_band = close - 1.5 * atr
    
    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # need EMA, vol MA, and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper band, with volume, and 12h EMA trending up
            if (close[i] > upper_band[i] and vol_confirm[i] and 
                ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band, with volume, and 12h EMA trending down
            elif (close[i] < lower_band[i] and vol_confirm[i] and 
                  ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below upper band (failed breakout) or EMA trend changes
            if (close[i] < upper_band[i] or 
                ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above lower band (failed breakout) or EMA trend changes
            if (close[i] > lower_band[i] or 
                ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_TrueRange_Breakout_EMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0