#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 24-bar avg). 
Enters long when price breaks above R1 in 1d uptrend with volume spike, short when breaks below S1 in 1d downtrend with volume spike. 
Exits on ATR-based stoploss (2.5x ATR) or reverse Camarilla level touch. Uses discrete sizing (0.25) to limit fee churn. 
Designed for 12h timeframe with ~12-25 trades/year, works in bull/bear by following 1d trend filter and avoiding false breakouts via volume confirmation.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    # Camarilla levels based on previous day's range
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    
    # Calculate Camarilla levels for current day based on previous day's OHLC
    range_prev = prev_high - prev_low
    R1 = prev_close + range_prev * 1.1 / 12
    S1 = prev_close - range_prev * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (using same 1d -> 12h alignment)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # ATR for stoploss (using 12h data)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough data for indicators
    start_idx = max(34, 24, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above R1 in 1d uptrend with volume confirmation
            bullish_breakout = (curr_close > R1_aligned[i]) and \
                              (close_1d[i] > ema_34_1d_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below S1 in 1d downtrend with volume confirmation
            bearish_breakout = (curr_close < S1_aligned[i]) and \
                              (close_1d[i] < ema_34_1d_aligned[i]) and \
                              volume_spike[i]
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: stoploss hit OR price touches S1 (reverse level)
            if (curr_close <= entry_price - 2.5 * atr[i]) or \
               (curr_close < S1_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: stoploss hit OR price touches R1 (reverse level)
            if (curr_close >= entry_price + 2.5 * atr[i]) or \
               (curr_close > R1_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0