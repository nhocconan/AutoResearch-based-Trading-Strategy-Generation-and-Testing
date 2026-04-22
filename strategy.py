#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend + 1d Williams %R + volume confirmation
# Long when Supertrend up, Williams %R < -80 (oversold), volume spike
# Short when Supertrend down, Williams %R > -20 (overbought), volume spike
# Exit when Supertrend reverses or Williams %R returns to neutral range
# Supertrend captures trend direction, Williams %R identifies mean-reversion entries
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# Target: 15-30 trades/year with high win rate in trending markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate Supertrend on 1d data (ATR=10, multiplier=3.0)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3.0 * atr
    basic_lb = (high_1d + low_1d) / 2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    
    for i in range(len(close_1d)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if close_1d[i-1] <= final_ub[i-1]:
                final_ub[i] = min(basic_ub[i], final_ub[i-1])
            else:
                final_ub[i] = basic_ub[i]
                
            if close_1d[i-1] >= final_lb[i-1]:
                final_lb[i] = max(basic_lb[i], final_lb[i-1])
            else:
                final_lb[i] = basic_lb[i]
    
    # Supertrend
    supertrend = np.zeros_like(close_1d)
    trend_up = np.ones_like(close_1d, dtype=bool)  # True for uptrend
    
    for i in range(len(close_1d)):
        if i == 0:
            supertrend[i] = final_ub[i]
            trend_up[i] = True
        else:
            if trend_up[i-1]:
                if close_1d[i] <= final_ub[i]:
                    trend_up[i] = False
                    supertrend[i] = final_lb[i]
                else:
                    trend_up[i] = True
                    supertrend[i] = final_ub[i]
            else:
                if close_1d[i] >= final_lb[i]:
                    trend_up[i] = True
                    supertrend[i] = final_ub[i]
                else:
                    trend_up[i] = False
                    supertrend[i] = final_lb[i]
    
    # Align to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(supertrend_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        williams_r_val = williams_r_aligned[i]
        supertrend_val = supertrend_aligned[i]
        trend_up_val = trend_up_aligned[i] > 0.5  # Convert back to boolean
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Supertrend up, Williams %R < -80 (oversold), volume spike
            if trend_up_val and williams_r_val < -80 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Supertrend down, Williams %R > -20 (overbought), volume spike
            elif not trend_up_val and williams_r_val > -20 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Supertrend reverses or Williams %R returns to neutral
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when trend turns down or Williams %R exceeds -20
                if not trend_up_val or williams_r_val > -20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when trend turns up or Williams %R goes below -80
                if trend_up_val or williams_r_val < -80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Supertrend_WilliamsR_Volume"
timeframe = "6h"
leverage = 1.0