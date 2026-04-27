# 6h CCI Trend-Following with Volume Filter and Trend Confirmation
# Hypothesis: CCI captures cyclical price extremes. In trending markets, CCI > +100 signals strong uptrend continuation,
# CCI < -100 signals strong downtrend continuation. Combined with 1-day EMA trend filter and volume confirmation,
# this should work in both bull (riding trends) and bear (catching sharp declines) markets.
# Target: 20-50 trades/year on 6h timeframe to avoid fee drag.

#!/usr/bin/env python3
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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Exponential Moving Average (34-period) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * multiplier) + (ema_34_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period Commodity Channel Index (CCI) on 6h data
    # CCI = (Typical Price - SMA(TP,20)) / (0.015 * Mean Deviation)
    typical_price = (high + low + close) / 3.0
    cci = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(19, n):
            tp_slice = typical_price[i-19:i+1]
            sma_tp = np.mean(tp_slice)
            mean_dev = np.mean(np.abs(tp_slice - sma_tp))
            if mean_dev > 0:
                cci[i] = (typical_price[i] - sma_tp) / (0.015 * mean_dev)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(cci[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume filter: at least 1.3x average volume to avoid low-volume false signals
        vol_filter = vol_ratio > 1.3
        
        if position == 0:
            # Long: CCI > +100 (overbought/strong uptrend) AND price above 1-day EMA (uptrend filter)
            if cci[i] > 100 and price > ema_34_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: CCI < -100 (oversold/strong downtrend) AND price below 1-day EMA (downtrend filter)
            elif cci[i] < -100 and price < ema_34_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: CCI falls below +50 (weakening momentum) OR price crosses below 1-day EMA
            if cci[i] < 50 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: CCI rises above -50 (weakening momentum) OR price crosses above 1-day EMA
            if cci[i] > -50 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_CCI_Trend_Following_Volume"
timeframe = "6h"
leverage = 1.0