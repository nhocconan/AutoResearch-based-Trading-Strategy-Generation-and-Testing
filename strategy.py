#!/usr/bin/env python3
# 6h_cci_mean_reversion_v1
# Hypothesis: 6h mean reversion strategy using CCI(20) extremes (>100 long entry, <-100 short entry) with volume confirmation (>1.5x 20-bar avg volume) and trend alignment via 12h EMA(50). Enters long when CCI crosses above -100 from below with volume and price > 12h EMA(50); enters short when CCI crosses below 100 from above with volume and price < 12h EMA(50). Exits on CCI crossing zero (mean reversion completion). Uses discrete sizing (0.25) to limit fee churn. Target: 12-37 trades/year (50-150 total over 4 years). CCI captures overbought/oversold conditions; volume confirms reversal conviction; 12h EMA filters counter-trend noise in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # CCI calculation (20-period)
    tp = (high + low + close) / 3  # Typical Price
    tp_s = pd.Series(tp)
    ma_tp = tp_s.rolling(window=20, min_periods=20).mean().values
    mad = tp_s.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci = (tp - ma_tp) / (0.015 * mad)
    
    # Multi-timeframe: 12h EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    ema_50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filters
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: CCI crosses above zero (mean reversion complete)
            if cci[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses below zero (mean reversion complete)
            if cci[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for CCI mean reversion entry with volume and trend alignment
            # Long: CCI crosses above -100 from below
            cci_long_signal = (cci[i-1] <= -100) and (cci[i] > -100) and volume_confirmed and uptrend
            # Short: CCI crosses below 100 from above
            cci_short_signal = (cci[i-1] >= 100) and (cci[i] < 100) and volume_confirmed and downtrend
            
            if cci_long_signal:
                position = 1
                signals[i] = 0.25
            elif cci_short_signal:
                position = -1
                signals[i] = -0.25
    
    return signals