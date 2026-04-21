#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Supertrend + Volume Spike
# Long when Williams %R < -80 (oversold) + price > Supertrend + volume > 1.5x 20-day avg
# Short when Williams %R > -20 (overbought) + price < Supertrend + volume > 1.5x 20-day avg
# Williams %R identifies reversals, Supertrend filters direction, volume confirms conviction
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets
# Target: 15-25 trades/year by requiring all three conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d Supertrend (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + multiplier * atr
    basic_lb = (high_1d + low_1d) / 2 - multiplier * atr
    
    # Final Supertrend
    final_ub = np.zeros(len(close_1d))
    final_lb = np.zeros(len(close_1d))
    supertrend = np.zeros(len(close_1d))
    trend = np.ones(len(close_1d))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]) else final_ub[i-1]
        final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]) else final_lb[i-1]
        
        if i == 1:
            supertrend[i] = final_ub[i]
            trend[i] = 1
        else:
            if supertrend[i-1] == final_ub[i-1]:
                supertrend[i] = final_lb[i] if close_1d[i] > final_lb[i] else final_ub[i]
                trend[i] = -1 if supertrend[i] == final_ub[i] else 1
            else:
                supertrend[i] = final_ub[i] if close_1d[i] < final_ub[i] else final_lb[i]
                trend[i] = 1 if supertrend[i] == final_lb[i] else -1
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        st = supertrend_aligned[i]
        tr = trend_aligned[i]
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Get current 1d volume (for volume confirmation)
        # Each 6h bar = 1/4 of a 1d bar, so we need to look at the current 1d bar
        idx_1d = i // 4
        if idx_1d >= len(df_1d):
            idx_1d = len(df_1d) - 1
        volume = df_1d['volume'].iloc[idx_1d] if idx_1d >= 0 else df_1d['volume'].iloc[0]
        volume_confirm = volume > 1.5 * vol_ma if vol_ma > 0 else False
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), price > Supertrend (uptrend), volume confirmation
            if wr < -80 and price > st and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price < Supertrend (downtrend), volume confirmation
            elif wr > -20 and price < st and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R > -50 (exiting oversold) or price crosses below Supertrend
                if wr > -50 or price < st:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R < -50 (exiting overbought) or price crosses above Supertrend
                if wr < -50 or price > st:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Supertrend_Volume"
timeframe = "6h"
leverage = 1.0