#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 12h EMA50 trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) with volume > 1.5x 20-bar average and close > 12h EMA50 (uptrend)
# Short when Williams %R > -20 (overbought) with volume > 1.5x 20-bar average and close < 12h EMA50 (downtrend)
# Exit when Williams %R reverses (> -50 for long, < -50 for short) or trend fails
# Williams %R identifies extreme reversals; works in bull (buy dips) and bear (sell rallies)
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "4h_WilliamsR_Volume_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 14, 20) + 1  # EMA50(12h) + Williams %R(14) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) with volume spike and close > 12h EMA50 (uptrend)
            if (williams_r[i] < -80 and 
                volume_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) with volume spike and close < 12h EMA50 (downtrend)
            elif (williams_r[i] > -20 and 
                  volume_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (reversing from oversold) or close < 12h EMA50 (trend failure)
            if (williams_r[i] > -50 or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (reversing from overbought) or close > 12h EMA50 (trend failure)
            if (williams_r[i] < -50 or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals