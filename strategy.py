#!/usr/bin/env python3
"""
1h RSI Pullback with 4h Trend and Volume Confirmation
Hypothesis: In trending markets (as defined by 4h EMA20), pullbacks to RSI(14) < 30 (long) or > 70 (short)
on 1h timeframe offer high-probability entries. Volume spike confirms institutional interest.
This works in both bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend).
Target: 15-37 trades/year (60-150 total over 4 years) with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(20) for trend
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns down
            if rsi_values[i] > 70 or close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns up
            if rsi_values[i] < 30 or close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Trend filter
            uptrend = close[i] > ema_20_4h_aligned[i]
            downtrend = close[i] < ema_20_4h_aligned[i]
            
            # Long: RSI < 30 (oversold) in uptrend + volume spike
            if (rsi_values[i] < 30 and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short: RSI > 70 (overbought) in downtrend + volume spike
            elif (rsi_values[i] > 70 and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals