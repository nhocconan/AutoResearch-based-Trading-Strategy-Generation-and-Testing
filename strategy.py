# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze + weekly pivot trend filter + volume confirmation
# - Bollinger Band width (BBW) percentile filter identifies low volatility regimes (squeeze)
# - Weekly pivot (from 1w data) provides directional bias for breakouts
# - Volume confirmation ensures breakout authenticity
# - Works in both bull/bear markets: squeeze identifies compression before explosive moves
# - Weekly pivot avoids counter-trend trades in strong trends
# Target: 15-30 trades/year (60-120 total over 4 years)
# Size: 0.25 (discrete levels to minimize churn)

name = "6h_1w_BollingerSqueeze_PivotTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    def calculate_weekly_pivot(high_arr, low_arr, close_arr):
        n_weeks = len(close_arr)
        PP = np.full(n_weeks, np.nan)
        R1 = np.full(n_weeks, np.nan)
        S1 = np.full(n_weeks, np.nan)
        
        for i in range(1, n_weeks):
            # Use previous week's OHLC
            high_prev = high_arr[i-1]
            low_prev = low_arr[i-1]
            close_prev = close_arr[i-1]
            
            # Standard pivot point calculation
            PP[i] = (high_prev + low_prev + close_prev) / 3.0
            R1[i] = 2 * PP[i] - low_prev
            S1[i] = 2 * PP[i] - high_prev
        
        return PP, R1, S1
    
    PP_1w, R1_1w, S1_1w = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    
    # Align weekly pivot to 6h timeframe
    PP_1w_aligned = align_htf_to_ltf(prices, df_1w, PP_1w)
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    
    # Calculate Bollinger Bands on daily data (20-period, 2 std dev)
    close_series = pd.Series(close_1d)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band width percentile (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(
        window=50, min_periods=50
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False).values
    
    # Align Bollinger width percentile to 6h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Calculate volume spike (volume > 1.8 * 30-period average)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(PP_1w_aligned[i]) or np.isnan(R1_1w_aligned[i]) or 
            np.isnan(S1_1w_aligned[i]) or np.isnan(bb_width_percentile_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Bollinger squeeze condition: BBW percentile < 20 (low volatility)
        squeeze_condition = bb_width_percentile_aligned[i] < 20
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when: squeeze breaks above weekly R1 with volume
            if (squeeze_condition and vol_confirm and 
                close[i] > R1_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short when: squeeze breaks below weekly S1 with volume
            elif (squeeze_condition and vol_confirm and 
                  close[i] < S1_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below weekly PP (mean reversion)
            if close[i] < PP_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above weekly PP (mean reversion)
            if close[i] > PP_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals