#/usr/bin/env python3
"""
4h_Volume_Spike_Momentum_12hTrend_Filter
Hypothesis: Volume spikes confirm momentum in the direction of 12h trend. Works in bull/bear via trend filter.
Target: 25-40 trades/year on 4h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h trend: EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 2.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    # Price momentum: close > open (bullish candle) or close < open (bearish candle)
    bullish_momentum = close > prices['open'].values
    bearish_momentum = close < prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        bull_mom = bullish_momentum[i]
        bear_mom = bearish_momentum[i]
        
        if position == 0:
            # Long: volume spike + bullish candle + uptrend
            if vol_spike_val and bull_mom and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: volume spike + bearish candle + downtrend
            elif vol_spike_val and bear_mom and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend turns down or volume spike in opposite direction
            if close[i] < ema_trend or (vol_spike_val and bear_mom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend turns up or volume spike in opposite direction
            if close[i] > ema_trend or (vol_spike_val and bull_mom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Volume_Spike_Momentum_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0