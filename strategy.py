#!/usr/bin/env python3
# 4h_volume_price_action_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter.
# Long when price breaks above Donchian high with volume spike and 1d uptrend.
# Short when price breaks below Donchian low with volume spike and 1d downtrend.
# Uses ATR-based stop loss to limit drawdown. Designed for 20-50 trades/year on 4h.
# Works in bull/bear via volume confirmation and trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_price_action_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i+1])
        donchian_low[i] = np.min(low[i-20:i+1])
    
    # Volume moving average (20-period) for spike detection
    volume_ma = np.full(n, np.nan)
    for i in range(20, n):
        volume_ma[i] = np.mean(volume[i-20:i+1])
    
    # ATR (14-period) for stop loss and filtering
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i+1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * volume_ma[i]
        
        # 1d trend filter
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or ATR-based stop
            if close[i] < donchian_low[i] or close[i] < donchian_high[i] - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or ATR-based stop
            if close[i] > donchian_high[i] or close[i] > donchian_low[i] + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume spike and 1d uptrend
            if (close[i] > donchian_high[i] and 
                volume_spike and 
                uptrend_1d):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume spike and 1d downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_spike and 
                  downtrend_1d):
                position = -1
                signals[i] = -0.25
    
    return signals