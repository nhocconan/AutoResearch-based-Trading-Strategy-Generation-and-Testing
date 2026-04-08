#!/usr/bin/env python3
# 6h_12h_1d_cci_trend_reversal_v1
# Hypothesis: 6h CCI(20) extreme readings (>100 or <-100) with 12h trend filter (EMA50) and 1d volume confirmation.
# Long when CCI < -100 (oversold) and 12h EMA50 rising and 1d volume > 1.5x average.
# Short when CCI > 100 (overbought) and 12h EMA50 falling and 1d volume > 1.5x average.
# Uses 1d volume spike to confirm reversals in overextended moves. Designed for 15-35 trades/year on 6h.
# Works in bull/bear via mean reversion in overextended conditions with trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_cci_trend_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3
    # Calculate 20-period SMA of typical price
    sma_tp = np.full(n, np.nan)
    for i in range(19, n):
        sma_tp[i] = np.mean(typical_price[i-19:i+1])
    # Calculate mean deviation
    mad = np.full(n, np.nan)
    for i in range(19, n):
        mad[i] = np.mean(np.abs(typical_price[i-19:i+1] - sma_tp[i]))
    # CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
    cci = np.full(n, np.nan)
    for i in range(19, n):
        if mad[i] != 0:
            cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mad[i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    # 12h EMA50 slope (rising/falling)
    ema50_slope = np.full(len(ema50_12h_aligned), np.nan)
    for i in range(1, len(ema50_12h_aligned)):
        if not np.isnan(ema50_12h_aligned[i]) and not np.isnan(ema50_12h_aligned[i-1]):
            ema50_slope[i] = ema50_12h_aligned[i] - ema50_12h_aligned[i-1]
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # 1d volume average (20-period)
    vol_avg_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(19, 50)  # Ensure CCI and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema50_slope[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x average
        volume_spike = volume_1d[i] > 1.5 * vol_avg_1d_aligned[i] if i < len(volume_1d) else False
        
        if position == 1:  # Long position
            # Exit: CCI returns above -50 or 12h EMA50 starts falling
            if cci[i] > -50 or ema50_slope[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI returns below 50 or 12h EMA50 starts rising
            if cci[i] < 50 or ema50_slope[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI < -100 (oversold), 12h EMA50 rising, volume spike
            if (cci[i] < -100 and 
                ema50_slope[i] > 0 and 
                volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: CCI > 100 (overbought), 12h EMA50 falling, volume spike
            elif (cci[i] > 100 and 
                  ema50_slope[i] < 0 and 
                  volume_spike):
                position = -1
                signals[i] = -0.25
    
    return signals