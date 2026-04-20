#!/usr/bin/env python3
"""
1d_VWAP_Deviation_MeanReversion
Hypothesis: Trade mean-reversion from VWAP on daily timeframe with weekly trend filter.
Long when price deviates below VWAP by 1.5 standard deviations in weekly uptrend.
Short when price deviates above VWAP by 1.5 standard deviations in weekly downtrend.
Uses volume-weighted average price (VWAP) as dynamic mean and Bollinger-like bands for entry.
Weekly trend filter (EMA50) avoids counter-trend trades. Target: 10-25 trades/year.
Works in bull/bear: weekly trend filter captures major moves, VWAP deviation captures mean reversion within trend.
"""

name = "1d_VWAP_Deviation_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price for VWAP calculation
    typical_price = (high + low + close) / 3.0
    
    # Calculate VWAP and standard deviation for the day
    vwap = np.full(n, np.nan)
    vwap_std = np.full(n, np.nan)
    
    # Group by date for daily VWAP calculation
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    for date in unique_dates:
        mask = (dates == date)
        if not np.any(mask):
            continue
        
        # Calculate cumulative VWAP for the day
        cum_vol = np.cumsum(volume[mask])
        cum_tp_vol = np.cumsum(typical_price[mask] * volume[mask])
        
        # Avoid division by zero
        vwap_day = np.where(cum_vol > 0, cum_tp_vol / cum_vol, typical_price[mask])
        
        # Calculate standard deviation of typical price from VWAP
        squared_diff = (typical_price[mask] - vwap_day) ** 2
        var_day = np.cumsum(squared_diff * volume[mask]) / np.where(cum_vol > 0, cum_vol, 1)
        std_day = np.sqrt(var_day)
        
        vwap[mask] = vwap_day
        vwap_std[mask] = std_day
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1w = ema(close_1w, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Define entry thresholds: 1.5 standard deviations from VWAP
    upper_band = vwap + (1.5 * vwap_std)
    lower_band = vwap - (1.5 * vwap_std)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below lower band AND weekly uptrend (price > EMA50)
            if close[i] < lower_band[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above upper band AND weekly downtrend (price < EMA50)
            elif close[i] > upper_band[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP OR weekly trend turns down
            if close[i] > vwap[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP OR weekly trend turns up
            if close[i] < vwap[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals