#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d RSI extremes with 1w Supertrend trend filter and volume confirmation
# Long when 1d RSI < 30 (oversold) AND 1w Supertrend = bullish (price > Supertrend) AND volume > 1.8 * avg_volume(20) on 12h
# Short when 1d RSI > 70 (overbought) AND 1w Supertrend = bearish (price < Supertrend) AND volume > 1.8 * avg_volume(20) on 12h
# Exit when 1d RSI crosses back through 50 (mean reversion to midpoint)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# RSI extremes provide high-probability reversal points in ranging markets
# 1w Supertrend filter ensures we trade with the dominant weekly trend
# Volume spike confirmation (1.8x) validates reversal strength while limiting overtrading
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets

name = "12h_1dRSI_Extreme_1wSupertrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed 1d bars for RSI
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    # Handle division by zero (when avg_loss == 0)
    rsi_1d = np.where(avg_loss.values == 0, 100, rsi_1d)
    rsi_1d = np.where(np.isnan(rsi_1d), 50, rsi_1d)
    
    # Align 1d RSI to 12h timeframe (wait for completed 1d bar)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data ONCE before loop for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for Supertrend
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend(10, 3.0)
    atr_period = 10
    atr_mult = 3.0
    
    # True Range
    tr1 = pd.Series(high_1w) - pd.Series(low_1w)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()
    
    # Basic Upper and Lower Bands
    basic_ub = (pd.Series(high_1w) + pd.Series(low_1w)) / 2 + atr_mult * atr
    basic_lb = (pd.Series(high_1w) + pd.Series(low_1w)) / 2 - atr_mult * atr
    
    # Final Upper and Lower Bands
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    for i in range(1, len(basic_ub)):
        if basic_ub.iloc[i] < final_ub.iloc[i-1] or close_1w.iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]
            
        if basic_lb.iloc[i] > final_lb.iloc[i-1] or close_1w.iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]
    
    # Supertrend
    supertrend = pd.Series(index=close_1w.index, dtype=float)
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i] = 0.0  # undefined
        elif supertrend.iloc[i-1] == final_ub.iloc[i-1]:
            supertrend.iloc[i] = final_ub.iloc[i] if close_1w.iloc[i] <= final_ub.iloc[i] else final_lb.iloc[i]
        else:
            supertrend.iloc[i] = final_lb.iloc[i] if close_1w.iloc[i] >= final_lb.iloc[i-1] else final_ub.iloc[i]
    
    # Supertrend trend: 1 = bullish (price > Supertrend), -1 = bearish (price < Supertrend)
    supertrend_trend = np.where(close_1w > supertrend.values, 1, -1)
    supertrend_trend = np.where(np.isnan(supertrend_trend), 0, supertrend_trend)
    
    # Align 1w Supertrend trend to 12h timeframe (wait for completed 1w bar)
    supertrend_trend_aligned = align_htf_to_ltf(prices, df_1w, supertrend_trend)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi_aligned[i]) or np.isnan(supertrend_trend_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold), Supertrend bullish, volume spike, in session
            if (rsi_aligned[i] < 30 and 
                supertrend_trend_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought), Supertrend bearish, volume spike, in session
            elif (rsi_aligned[i] > 70 and 
                  supertrend_trend_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses back above 50 (mean reversion)
            if rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses back below 50 (mean reversion)
            if rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals