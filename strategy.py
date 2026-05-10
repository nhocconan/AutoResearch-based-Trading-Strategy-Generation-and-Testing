#!/usr/bin/env python3
# 4h_CCI_Trend_Follow_1dTrend_Volume
# Hypothesis: Use CCI(20) on 4h for momentum signals, filtered by 1d EMA trend and volume spikes.
# CCI captures cyclical moves and reversals, while 1d EMA ensures alignment with higher-timeframe trend.
# Volume confirmation adds conviction to breakouts. Designed for 20-30 trades/year to avoid fee drag.
# Works in bull markets via trend following and in bear markets via mean-reversion in extremes.

name = "4h_CCI_Trend_Follow_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # CCI(20) calculation
    cci_period = 20
    tp = (high + low + close) / 3
    sma_tp = pd.Series(tp).rolling(window=cci_period, min_periods=cci_period).mean().values
    mad = pd.Series(tp).rolling(window=cci_period, min_periods=cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CCI crosses above -100 (bullish momentum), price above 1d EMA, volume confirmation
            if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses below 100 (bearish momentum), price below 1d EMA, volume confirmation
            elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: CCI crosses below +100 (overbought) or trend change
            if cci[i] < 100 and cci[i-1] >= 100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: CCI crosses above -100 (oversold) or trend change
            if cci[i] > -100 and cci[i-1] <= -100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals