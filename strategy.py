#!/usr/bin/env python3
"""
1h_4h_1d_Trend_With_Volume_Filter
Hypothesis: Use 4h trend (EMA21 > EMA50) and 1d momentum (close > SMA50) as directional filters, with 1h RSI(14) pullback entries (RSI < 40 for longs, RSI > 60 for shorts) and volume confirmation (volume > 1.5x 20-period average). This avoids counter-trend trades, captures pullbacks in strong trends, and reduces whipsaws. Volume filter ensures trades occur with participation. Targets 15-30 trades/year by requiring multiple confluence factors, with position size 0.20.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h indicators
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # EMA21 and EMA50 on 4h
    ema21_4h = np.full_like(close_4h, np.nan)
    ema50_4h = np.full_like(close_4h, np.nan)
    
    # EMA21
    alpha21 = 2 / (21 + 1)
    for i in range(len(close_4h)):
        if i == 0:
            ema21_4h[i] = close_4h[i]
        elif np.isnan(ema21_4h[i-1]):
            ema21_4h[i] = close_4h[i]
        else:
            ema21_4h[i] = alpha21 * close_4h[i] + (1 - alpha21) * ema21_4h[i-1]
    
    # EMA50
    alpha50 = 2 / (50 + 1)
    for i in range(len(close_4h)):
        if i == 0:
            ema50_4h[i] = close_4h[i]
        elif np.isnan(ema50_4h[i-1]):
            ema50_4h[i] = close_4h[i]
        else:
            ema50_4h[i] = alpha50 * close_4h[i] + (1 - alpha50) * ema50_4h[i-1]
    
    # Get 1d data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # SMA50 on 1d
    sma50_1d = np.full_like(close_1d, np.nan)
    for i in range(50, len(close_1d)):
        sma50_1d[i] = np.mean(close_1d[i-50:i])
    
    # Align 4h and 1d indicators to 1h
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # need SMA50_1d, vol_ma, rsi
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(sma50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: 4h uptrend (EMA21 > EMA50), 1d momentum (close > SMA50), 
            # 1h RSI pullback (<40), and volume confirmation
            if (ema21_4h_aligned[i] > ema50_4h_aligned[i] and 
                close[i] > sma50_1d_aligned[i] and 
                rsi[i] < 40 and 
                volume_filter):
                signals[i] = 0.20
                position = 1
            # Short entry: 4h downtrend (EMA21 < EMA50), 1d momentum (close < SMA50), 
            # 1h RSI pullback (>60), and volume confirmation
            elif (ema21_4h_aligned[i] < ema50_4h_aligned[i] and 
                  close[i] < sma50_1d_aligned[i] and 
                  rsi[i] > 60 and 
                  volume_filter):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: 4h trend changes (EMA21 < EMA50) or RSI overbought (>70)
            if (ema21_4h_aligned[i] < ema50_4h_aligned[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend changes (EMA21 > EMA50) or RSI oversold (<30)
            if (ema21_4h_aligned[i] > ema50_4h_aligned[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Trend_With_Volume_Filter"
timeframe = "1h"
leverage = 1.0