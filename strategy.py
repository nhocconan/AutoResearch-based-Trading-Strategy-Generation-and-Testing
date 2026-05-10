#!/usr/bin/env python3
# 6h_LiquiditySweep_1dOrderBlock_WeeklyTrend
# Hypothesis: Combines liquidity sweeps (fake breakouts) with 1-day order blocks and weekly trend filter.
# Long: Price sweeps below prior low then closes back above it (bullish trap), with bullish 1-day order block and weekly uptrend.
# Short: Price sweeps above prior high then closes back below it (bearish trap), with bearish 1-day order block and weekly downtrend.
# Uses volume confirmation to validate the reversal after sweep.
# Designed to work in both bull and bear markets by trapping liquidity before continuation.
# Targets 12-30 trades per year on 6h timeframe with position size 0.25.

name = "6h_LiquiditySweep_1dOrderBlock_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for order blocks and weekly data for trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1-week EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Identify 1-day bullish and bearish order blocks
    # Bullish OB: last down candle before up move (close < open, then next candle closes above its high)
    # Bearish OB: last up candle before down move (close > open, then next candle closes below its low)
    bullish_ob = np.zeros(len(df_1d), dtype=bool)
    bearish_ob = np.zeros(len(df_1d), dtype=bool)
    
    for i in range(1, len(df_1d)):
        # Bullish OB: current candle is up, previous was down and we broke above its high
        if (df_1d['close'].iloc[i] > df_1d['open'].iloc[i] and  # current up
            df_1d['close'].iloc[i-1] < df_1d['open'].iloc[i-1] and  # previous down
            df_1d['close'].iloc[i] > df_1d['high'].iloc[i-1]):  # broke above prev high
            bullish_ob[i-1] = True  # mark the down candle as bullish OB
        
        # Bearish OB: current candle is down, previous was up and we broke below its low
        if (df_1d['close'].iloc[i] < df_1d['open'].iloc[i] and  # current down
            df_1d['close'].iloc[i-1] > df_1d['open'].iloc[i-1] and  # previous up
            df_1d['close'].iloc[i] < df_1d['low'].iloc[i-1]):  # broke below prev low
            bearish_ob[i-1] = True  # mark the up candle as bearish OB
    
    # Align order blocks to 6h
    bullish_ob_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob.astype(float))
    bearish_ob_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob.astype(float))
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long setup: liquidity sweep below prior low + bullish OB + weekly uptrend
            if i > 0:
                # Sweep below prior low then close back above it (bullish trap)
                swept_below = low[i] < low[i-1]
                closed_back = close[i] > low[i-1]
                bullish_trap = swept_below and closed_back
                
                if (bullish_trap and 
                    bullish_ob_aligned[i] > 0.5 and  # bullish OB present
                    volume_confirm[i] and
                    weekly_uptrend):
                    signals[i] = 0.25
                    position = 1
            
            # Short setup: liquidity sweep above prior high + bearish OB + weekly downtrend
            elif i > 0:
                # Sweep above prior high then close back below it (bearish trap)
                swept_above = high[i] > high[i-1]
                closed_back = close[i] < high[i-1]
                bearish_trap = swept_above and closed_back
                
                if (bearish_trap and 
                    bearish_ob_aligned[i] > 0.5 and  # bearish OB present
                    volume_confirm[i] and
                    weekly_downtrend):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price breaks below the swept low or weekly trend fails
            if i > 0 and (low[i] < low[i-1] or not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above the swept high or weekly trend fails
            if i > 0 and (high[i] > high[i-1] or not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals