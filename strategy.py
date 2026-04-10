#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d EMA200 trend filter and volume confirmation
# - Camarilla pivot levels from 1d: calculates H3/L3 levels from previous day's range
# - Breakout above H3 or below L3 with volume confirmation and 1d trend alignment
# - EMA200 trend filter ensures we trade with higher timeframe trend
# - Volume confirmation: current volume > 1.5x 20-period average to avoid false breakouts
# - Exit: opposite Camarilla level touch (H3 for longs, L3 for shorts)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 20-40 trades/year on 4h (80-160 total over 4 years) to minimize fee drag
# - Works in both bull/bear: EMA200 trend filter adapts to regime, volume confirmation reduces whipsaws

name = "4h_1d_camarilla_pivot_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute 1d Camarilla pivot levels (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels: H3 = close + range * 1.1/4, L3 = close - range * 1.1/4
    h3 = close_1d + range_hl * 1.1 / 4.0
    l3 = close_1d - range_hl * 1.1 / 4.0
    
    # Align HTF levels to LTF
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Get current 1d close for trend filter (aligned)
        close_1d_current = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        
        # 1d trend filter: price > EMA200 = bullish, price < EMA200 = bearish
        bullish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] > trend_aligned[i]
        bearish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > H3 AND bullish trend AND volume confirmation
            if prices['close'].iloc[i] > h3_aligned[i] and bullish_trend and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: price < L3 AND bearish trend AND volume confirmation
            elif prices['close'].iloc[i] < l3_aligned[i] and bearish_trend and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price touches opposite Camarilla level
            exit_long = prices['close'].iloc[i] < l3_aligned[i]   # Price breaks below L3 (exit long)
            exit_short = prices['close'].iloc[i] > h3_aligned[i]  # Price breaks above H3 (exit short)
            
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals