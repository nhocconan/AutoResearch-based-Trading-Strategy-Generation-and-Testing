#!/usr/bin/env python3
"""
4h_Williams_VIX_Fix_MeanReversion_1dTrendFilter
Hypothesis: Williams VIX Fix identifies volatility spikes and mean reversion opportunities in 4h timeframe. 
Long when VIX Fix > upper band AND price < 1d EMA200 (oversold in uptrend).
Short when VIX Fix > upper band AND price > 1d EMA200 (overbought in downtrend).
Exit when VIX Fix < middle band or opposite condition met.
Uses 1d trend filter to avoid counter-trend trades in strong markets. Designed for ~80-150 trades over 4 years (20-38/year) via volatility mean reversion with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # need 200 for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Williams VIX Fix calculation (22-period)
    # VIX Fix = ((Highest Close in period - Low) / (Highest Close in period)) * 100
    vixfix_period = 22
    highest_close = pd.Series(close).rolling(window=vixfix_period, min_periods=vixfix_period).max().values
    vixfix = ((highest_close - low) / highest_close) * 100
    
    # Bollinger Bands on VIX Fix (20-period, 2 std dev)
    vixfix_ma = pd.Series(vixfix).rolling(window=20, min_periods=20).mean().values
    vixfix_std = pd.Series(vixfix).rolling(window=20, min_periods=20).std().values
    upper_band = vixfix_ma + (2 * vixfix_std)
    middle_band = vixfix_ma
    lower_band = vixfix_ma - (2 * vixfix_std)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, vixfix_period, 20, 200)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(vixfix[i]) or 
            np.isnan(upper_band[i]) or np.isnan(middle_band[i]) or np.isnan(lower_band[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_200_1d_aligned[i]
        vixfix_val = vixfix[i]
        upper = upper_band[i]
        middle = middle_band[i]
        lower = lower_band[i]
        
        if position == 0:
            # Long: VIX Fix above upper band (high volatility) AND price below 1d EMA200 (oversold in uptrend)
            if (vixfix_val > upper) and (close[i] < ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: VIX Fix above upper band (high volatility) AND price above 1d EMA200 (overbought in downtrend)
            elif (vixfix_val > upper) and (close[i] > ema_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when VIX Fix returns to middle band (mean reversion complete) OR flip to short signal
            if vixfix_val < middle or ((vixfix_val > upper) and (close[i] > ema_trend)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when VIX Fix returns to middle band (mean reversion complete) OR flip to long signal
            if vixfix_val < middle or ((vixfix_val > upper) and (close[i] < ema_trend)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Williams_VIX_Fix_MeanReversion_1dTrendFilter"
timeframe = "4h"
leverage = 1.0