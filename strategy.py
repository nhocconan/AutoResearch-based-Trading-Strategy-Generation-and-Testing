#!/usr/bin/env python3
"""
1h Volume Spike + 4h Trend + 1d Momentum Filter
Hypothesis: In 1h timeframe, volume spikes combined with 4h EMA trend alignment
and 1d RSI momentum filter captures high-probability moves while avoiding false signals.
Uses 4h for trend direction, 1d for momentum regime, 1h for precise entry timing.
Target: 15-37 trades/year (60-150 total over 4 years) with volume spike filter reducing noise.
Works in bull via long signals, in bear via short signals with trend/momentum alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_spike_4h_trend_1d_momentum_v1"
timeframe = "1h"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # RSI(14) on 1d
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 1h volume spike filter
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: 4h trend turns bearish or 1d momentum weakens
            if close[i] < ema_20_4h_aligned[i] or rsi_14_1d_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 4h trend turns bullish or 1d momentum strengthens
            if close[i] > ema_20_4h_aligned[i] or rsi_14_1d_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Trend alignment: price vs 4h EMA20
            uptrend_4h = close[i] > ema_20_4h_aligned[i]
            downtrend_4h = close[i] < ema_20_4h_aligned[i]
            
            # Momentum filter: 1d RSI
            bullish_momentum = rsi_14_1d_aligned[i] > 55
            bearish_momentum = rsi_14_1d_aligned[i] < 45
            
            # Long: volume spike + 4h uptrend + 1d bullish momentum
            if (vol_spike[i] and 
                uptrend_4h and 
                bullish_momentum):
                position = 1
                signals[i] = 0.20
            # Short: volume spike + 4h downtrend + 1d bearish momentum
            elif (vol_spike[i] and 
                  downtrend_4h and 
                  bearish_momentum):
                position = -1
                signals[i] = -0.20
    
    return signals