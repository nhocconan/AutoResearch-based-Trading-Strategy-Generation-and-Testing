#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action strategy using 1d Keltner Channel breakout with volume confirmation and 1d trend filter.
# Keltner Channel (based on ATR) adapts to volatility, providing dynamic support/resistance.
# Long when price breaks above upper KC with volume and 1d EMA up; short when breaks below lower KC with volume and 1d EMA down.
# Designed to capture breakouts in trending markets while avoiding whipsaw in ranging conditions.
# Works in both bull and bear markets by following 1d EMA trend direction.
name = "4h_KeltnerBreakout_1dEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Keltner Channel and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Keltner Channel: 20-period EMA of typical price ± 2 * ATR(10)
    typical_price = (high + low + close) / 3
    ema_20 = pd.Series(typical_price).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    
    # 1d EMA trend filter (50-period)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_20[i]) or np.isnan(atr_10[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper KC + volume + 1d EMA up
            if (price > kc_upper[i] and vol_confirm[i] and price > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC + volume + 1d EMA down
            elif (price < kc_lower[i] and vol_confirm[i] and price < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle line (EMA20) or reverses below lower KC
            if price < ema_20[i] or price < kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle line (EMA20) or reverses above upper KC
            if price > ema_20[i] or price > kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals