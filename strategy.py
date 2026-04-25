#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSp
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation (>2x 20-bar avg). Enters long when price breaks above R1 in 12h uptrend, short when breaks below S1 in 12h downtrend. Uses ATR-based stoploss (1.5x ATR) and discrete sizing (0.30) to limit fee churn. Designed for 4h timeframe with ~20-50 trades/year, works in bull/bear by following 12h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14 periods)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need 20-period data for volume MA and 50 for 12h EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Calculate Camarilla levels for current 4h bar using previous day's OHLC
        # Get the index of the 1d bar that corresponds to the current 4h period
        # We'll use the 1d data to get proper daily OHLC
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Align 1d data to 4h timeframe for proper Camarilla calculation
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
        low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
        
        # Use previous completed 1d bar for Camarilla calculation
        if i > 0:
            prev_close_1d = close_1d_aligned[i-1]
            prev_high_1d = high_1d_aligned[i-1]
            prev_low_1d = low_1d_aligned[i-1]
        else:
            prev_close_1d = close_1d_aligned[0]
            prev_high_1d = high_1d_aligned[0]
            prev_low_1d = low_1d_aligned[0]
        
        # Camarilla levels calculation
        range_1d = prev_high_1d - prev_low_1d
        camarilla_r1 = prev_close_1d + (range_1d * 1.1 / 12)
        camarilla_s1 = prev_close_1d - (range_1d * 1.1 / 12)
        
        if position == 0:
            # Long: price breaks above R1 in 12h uptrend with volume confirmation
            bullish_breakout = (curr_close > camarilla_r1) and \
                              (close[i] > ema_50_12h_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below S1 in 12h downtrend with volume confirmation
            bearish_breakout = (curr_close < camarilla_s1) and \
                              (close[i] < ema_50_12h_aligned[i]) and \
                              volume_spike[i]
            
            if bullish_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - (1.5 * atr[i])
            elif bearish_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + (1.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price breaks below S1 OR stoploss hit OR trend turns down
            if (curr_close < camarilla_s1) or \
               (curr_close < atr_stop) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price breaks above R1 OR stoploss hit OR trend turns up
            if (curr_close > camarilla_r1) or \
               (curr_close > atr_stop) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSp"
timeframe = "4h"
leverage = 1.0