#!/usr/bin/env python3
"""
1h EMA Pullback + Volume Spike + 4h Trend Filter
Trades pullbacks to 21 EMA in 4h trend direction with volume confirmation.
Designed for low trade frequency (15-30/year) with high win rate in trending markets.
Uses 4h EMA for trend direction, 1h EMA for entry timing, volume spike for confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA21 for trend direction
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1h EMA21 for entry timing
    ema_21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_21_1h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_4h = ema_21_4h_aligned[i]
        ema_1h = ema_21_1h[i]
        
        if position == 0:
            # Long: price above 4h EMA (uptrend) + pulls back to 1h EMA + volume spike
            if (price > ema_4h and 
                price <= ema_1h * 1.005 and  # within 0.5% above EMA (pullback)
                price >= ema_1h * 0.995 and  # within 0.5% below EMA
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA (downtrend) + pulls back to 1h EMA + volume spike
            elif (price < ema_4h and 
                  price <= ema_1h * 1.005 and  # within 0.5% above EMA (pullback)
                  price >= ema_1h * 0.995 and  # within 0.5% below EMA
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.20
            # Exit: price breaks below 1h EMA (trend weakness)
            if price < ema_1h:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.20
            # Exit: price breaks above 1h EMA (trend weakness)
            if price > ema_1h:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_EMA_Pullback_VolumeSpike_4hTrend"
timeframe = "1h"
leverage = 1.0