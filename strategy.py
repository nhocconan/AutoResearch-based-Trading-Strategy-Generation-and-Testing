#!/usr/bin/env python3
"""
1h Volatility Breakout with 4h Trend Filter and Volume Confirmation
Hypothesis: Breakouts from low volatility (ATR contraction) capture explosive moves.
When aligned with 4h trend (EMA50) and volume spikes, it avoids whipsaws in both bull and bear markets.
Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volatility_breakout_4h_trend_volume_v1"
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
    
    # 4h data for trend filter and volatility
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h ATR(14) for volatility measurement
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # 1h ATR(14) for breakout threshold
    tr1h = high - low
    tr2h = np.abs(high - np.roll(close, 1))
    tr3h = np.abs(low - np.roll(close, 1))
    trh = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    trh[0] = tr1h[0]
    atr_14_1h = pd.Series(trh).rolling(window=14, min_periods=14).mean().values
    
    # 1h volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or
            np.isnan(atr_14_1h[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions: price breaks ATR-based bands
        upper_band = close[i-1] + (0.5 * atr_14_1h[i])  # 0.5x ATR breakout
        lower_band = close[i-1] - (0.5 * atr_14_1h[i])
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA50 or ATR expansion reversal
            if (close[i] < ema_50_4h_aligned[i] or 
                close[i] < close[i-1] - (0.3 * atr_14_1h[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA50 or ATR expansion reversal
            if (close[i] > ema_50_4h_aligned[i] or 
                close[i] > close[i-1] + (0.3 * atr_14_1h[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Trend filter: price vs 4h EMA50
            uptrend = close[i] > ema_50_4h_aligned[i]
            downtrend = close[i] < ema_50_4h_aligned[i]
            
            # Volatility filter: current ATR < 0.7x 4h ATR (contraction)
            vol_contract = atr_14_1h[i] < (0.7 * atr_14_4h_aligned[i])
            
            # Long: upward breakout + uptrend + vol contraction + volume spike
            if (close[i] > upper_band and 
                uptrend and 
                vol_contract and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short: downward breakout + downtrend + vol contraction + volume spike
            elif (close[i] < lower_band and 
                  downtrend and 
                  vol_contract and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals