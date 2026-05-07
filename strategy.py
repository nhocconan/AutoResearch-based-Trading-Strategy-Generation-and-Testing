#!/usr/bin/env python3
name = "1h_Stochastic_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    trend_up = close > ema50_4h_aligned
    trend_down = close < ema50_4h_aligned
    
    # Stochastic Oscillator (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    k_percent = np.where(denominator != 0, ((close - lowest_low) / denominator) * 100, 50)
    
    # %D line (3-period SMA of %K)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Stochastic signals: %K crosses above/below %D
    stoch_cross_up = (k_percent > d_percent) & (np.roll(k_percent, 1) <= np.roll(d_percent, 1))
    stoch_cross_down = (k_percent < d_percent) & (np.roll(k_percent, 1) >= np.roll(d_percent, 1))
    
    # Volume filter: volume above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(14, 20, 50)  # Stochastic, volume, trend
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(k_percent[i]) or 
            np.isnan(d_percent[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Stochastic bullish cross in oversold (<20) + 4h uptrend + volume + session
            if (stoch_cross_up[i] and 
                k_percent[i] < 20 and 
                trend_up[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: Stochastic bearish cross in overbought (>80) + 4h downtrend + volume + session
            elif (stoch_cross_down[i] and 
                  k_percent[i] > 80 and 
                  trend_down[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Stochastic bearish cross OR trend turns down
            if stoch_cross_down[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Stochastic bullish cross OR trend turns up
            if stoch_cross_up[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Stochastic oscillator identifies momentum reversals in overbought/oversold conditions.
# Combined with 4h EMA50 trend filter, volume confirmation, and active session filter (08-20 UTC)
# to avoid low-liquidity periods. Long when %K crosses above %D below 20 in uptrend,
# short when %K crosses below %D above 80 in downtrend. Position size 0.20 limits risk.
# Designed for 1h timeframe with ~20-40 trades/year to minimize fee drag.
# Works in bull markets (captures pullbacks in uptrend) and bear markets (captures bounces in downtrend).