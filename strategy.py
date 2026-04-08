#!/usr/bin/env python3
# 6h_cci_breakout_1d_trend_volume
# Hypothesis: CCI(20) breakout on 6h with 1d EMA trend filter and volume confirmation.
# Long when CCI crosses above +100 with uptrend (price > 1d EMA50) and volume > 1.5x average.
# Short when CCI crosses below -100 with downtrend (price < 1d EMA50) and volume > 1.5x average.
# Exit when CCI returns to zero line (indicates trend exhaustion).
# Uses monthly trend filter for regime alignment and avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_breakout_1d_trend_volume"
timeframe = "6h"
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
    
    # Get daily and weekly data for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly EMA20 for regime filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate CCI(20) on 6h data
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below zero (trend exhaustion)
            if cci[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above zero (trend exhaustion)
            if cci[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Regime filter: only trade in direction of weekly trend
            uptrend_regime = close[i] > ema_20_1w_aligned[i]
            downtrend_regime = close[i] < ema_20_1w_aligned[i]
            
            # CCI breakout entries
            if (cci[i] > 100) and (cci[i-1] <= 100) and (close[i] > ema_50_1d_aligned[i]) and volume_ok and uptrend_regime:
                position = 1
                signals[i] = 0.25
            elif (cci[i] < -100) and (cci[i-1] >= -100) and (close[i] < ema_50_1d_aligned[i]) and volume_ok and downtrend_regime:
                position = -1
                signals[i] = -0.25
    
    return signals