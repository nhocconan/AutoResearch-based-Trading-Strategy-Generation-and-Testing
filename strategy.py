#!/usr/bin/env python3
"""
Hypothesis: 6-hour Volume-Weighted MACD with 12-hour Trend Filter and Volume Spike.
Long when VW-MACD crosses above signal line, 12h EMA50 rising, and volume spike.
Short when VW-MACD crosses below signal line, 12h EMA50 falling, and volume spike.
Exit when VW-MACD crosses back or 12h EMA50 reverses.
Designed for low trade frequency (<30 trades/year) by requiring VW-MACD crossover + trend + volume.
Uses volume-weighted price to reduce noise in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume-weighted close: (high + low + close) / 3 * volume, then normalize
    # Actually, VWAP-like: typical price * volume
    typical_price = (high + low + close) / 3.0
    vw_close = typical_price * volume
    
    # Calculate VW-MACD: MACD on volume-weighted close
    # Fast EMA(12), Slow EMA(26), Signal EMA(9)
    vw_close_series = pd.Series(vw_close)
    ema12 = vw_close_series.ewm(span=12, adjust=False, min_periods=12).values
    ema26 = vw_close_series.ewm(span=26, adjust=False, min_periods=26).values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).values
    macd_hist = macd_line - signal_line
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(macd_line[i]) or np.isnan(signal_line[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # MACD crossover signals
        macd_bullish_cross = (macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1])
        macd_bearish_cross = (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1])
        
        if position == 0:
            # Long: VW-MACD bullish cross, 12h EMA50 rising, volume spike
            if macd_bullish_cross and ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: VW-MACD bearish cross, 12h EMA50 falling, volume spike
            elif macd_bearish_cross and ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: VW-MACD crosses back or 12h EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: VW-MACD bearish cross or 12h EMA50 turns down
                if macd_bearish_cross or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: VW-MACD bullish cross or 12h EMA50 turns up
                if macd_bullish_cross or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_VW_MACD_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0