#!/usr/bin/env python3
"""
4h ATR Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Breakouts beyond ATR bands capture trend continuations in both bull and bear markets.
Filtered by 1d EMA trend and volume spikes to avoid false breakouts in ranging conditions.
Targets 20-50 trades/year to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(100) for trend filter (slower for more reliability)
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # ATR(14) for breakout bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Upper and lower bands: close ± 2.5 * ATR
    upper_band = close + 2.5 * atr
    lower_band = close - 2.5 * atr
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower band or trend reversal
            if close[i] < lower_band[i] or close[i] < ema_100_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper band or trend reversal
            if close[i] > upper_band[i] or close[i] > ema_100_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1d EMA100
            uptrend = close[i] > ema_100_1d_aligned[i]
            downtrend = close[i] < ema_100_1d_aligned[i]
            
            # Long: price breaks above upper band + uptrend + volume spike
            if close[i] > upper_band[i] and uptrend and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower band + downtrend + volume spike
            elif close[i] < lower_band[i] and downtrend and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals