#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Combines 1d Bollinger Band mean reversion with volume confirmation and trend filter on 4h.
# Works in bull markets by buying pullbacks to lower BB in uptrend, and in bear markets by selling rallies to upper BB in downtrend.
# Uses tight entry conditions to limit trades (<50/year) and avoid fee drag.
name = "4h_1d_bb_meanrev_trend_v1"
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
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    lower_bb = sma_20 - 2 * std_20
    upper_bb = sma_20 + 2 * std_20
    
    # Align BB to 4h timeframe
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Get 1w data for trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(lower_bb_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend from 1w EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Mean reversion signals with volume confirmation
        # Long: price touches or goes below lower BB in uptrend
        long_signal = close[i] <= lower_bb_aligned[i] and uptrend and volume_ok[i]
        # Short: price touches or goes above upper BB in downtrend
        short_signal = close[i] >= upper_bb_aligned[i] and downtrend and volume_ok[i]
        
        # Exit when price returns to SMA (mean reversion complete)
        exit_long = close[i] >= sma_20_aligned[i]
        exit_short = close[i] <= sma_20_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals