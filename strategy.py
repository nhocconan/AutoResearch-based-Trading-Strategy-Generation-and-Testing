#!/usr/bin/env python3
"""
1h EMA21 Bounce with 4h Trend and Volume Spike
Hypothesis: In trending markets, price pulls back to the 21 EMA on 1h before continuing.
Use 4h EMA50 for trend direction and volume spike for confirmation. Works in both bull
and bear markets by only trading in the direction of the 4h trend. Designed for 15-30
trades/year on 1h timeframe with low turnover to minimize fee drag.
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
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMA21 for bounce level
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_21[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50 = ema_50_4h_aligned[i]
        ema_21_val = ema_21[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price near EMA21, above 4h EMA50, with volume spike
            if price > ema_50 and abs(price - ema_21_val) < 0.5 * atr_val and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price near EMA21, below 4h EMA50, with volume spike
            elif price < ema_50 and abs(price - ema_21_val) < 0.5 * atr_val and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.20
            # Exit: price moves 1.5*ATR away from EMA21 or trend changes
            if price < ema_21_val - 1.5 * atr_val or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.20
            # Exit: price moves 1.5*ATR away from EMA21 or trend changes
            if price > ema_21_val + 1.5 * atr_val or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_EMA21_Bounce_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0