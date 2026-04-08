#!/usr/bin/env python3
# 12h_bollinger_breakout_1d_trend_volume_v1
# Hypothesis: Bollinger Band breakouts on 12h with volume confirmation and 1d EMA trend filter.
# In bull markets: buy upper band breakouts in uptrend. In bear markets: sell lower band breakouts in downtrend.
# Bollinger Bands adapt to volatility, making them effective across regimes.
# Volume confirmation reduces false breakouts. Trend filter ensures we trade with the higher timeframe trend.
# Target: 25-35 trades/year (~100-140 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_bollinger_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 12h
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i]) if np.mean(volume[i-20:i]) > 0 else 0
            vol_filter = volume[i] > 1.5 * vol_ma
        else:
            vol_filter = False
        
        if position == 1:  # Long position
            # Exit: price crosses below middle band OR trend turns against us
            if (close[i] < sma[i]) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle band OR trend turns against us
            if (close[i] > sma[i]) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with volume and uptrend
            if (close[i] > upper[i]) and vol_filter and (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume and downtrend
            elif (close[i] < lower[i]) and vol_filter and (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals