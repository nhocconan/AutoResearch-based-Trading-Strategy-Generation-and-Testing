#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily trading strategy using weekly Supertrend for trend direction and daily ATR-based breakout for entry.
# Weekly Supertrend defines bull/bear regime; daily ATR breakout captures momentum within the trend.
# Designed to work in both bull and bear markets by only trading in the direction of the weekly trend.
# Targets 10-25 trades/year with disciplined risk control via ATR-based exits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Supertrend (trend filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Supertrend (10, 3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + multiplier * atr_1w
    lower_band = hl2 - multiplier * atr_1w
    
    # Final Upper and Lower Bands
    final_upper = np.zeros_like(upper_band)
    final_lower = np.zeros_like(lower_band)
    for i in range(len(close_1w)):
        if i == 0:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
        else:
            if close_1w[i-1] <= final_upper[i-1]:
                final_upper[i] = min(upper_band[i], final_upper[i-1])
            else:
                final_upper[i] = upper_band[i]
            
            if close_1w[i-1] >= final_lower[i-1]:
                final_lower[i] = max(lower_band[i], final_lower[i-1])
            else:
                final_lower[i] = lower_band[i]
    
    # Supertrend direction
    supertrend_dir = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    for i in range(1, len(close_1w)):
        if close_1w[i] > final_upper[i-1]:
            supertrend_dir[i] = 1
        elif close_1w[i] < final_lower[i-1]:
            supertrend_dir[i] = -1
        else:
            supertrend_dir[i] = supertrend_dir[i-1]
            if supertrend_dir[i] == 1 and final_lower[i] < final_lower[i-1]:
                final_lower[i] = final_lower[i-1]
            if supertrend_dir[i] == -1 and final_upper[i] > final_upper[i-1]:
                final_upper[i] = final_upper[i-1]
    
    # Align weekly Supertrend to daily
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir)
    
    # Load daily data for ATR breakout
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14)
    atr_period_d = 14
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_d[0] = 0
    tr2_d[0] = 0
    tr3_d[0] = 0
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_1d = pd.Series(tr_d).rolling(window=atr_period_d, min_periods=atr_period_d).mean().values
    
    # Calculate daily moving average (20) for breakout reference
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(atr_1d[i]) or 
            np.isnan(ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        atr = atr_1d[i]
        ma = ma_20[i]
        trend = supertrend_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above MA(20) + 0.5*ATR in uptrend
            if trend == 1 and price > ma + 0.5 * atr:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below MA(20) - 0.5*ATR in downtrend
            elif trend == -1 and price < ma - 0.5 * atr:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions: ATR-based trailing stop
            exit_signal = False
            
            if position == 1:  # long position
                # Trailing stop: highest close since entry minus 2.0*ATR
                # We approximate highest close using rolling max of close
                # For simplicity, use close-based trailing stop from entry
                if price < entry_price - 2.0 * atr:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Trailing stop: lowest close since entry plus 2.0*ATR
                if price > entry_price + 2.0 * atr:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklySupertrend_DailyATRBreakout"
timeframe = "1d"
leverage = 1.0