#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1w EMA50 trend filter and volume regime filter.
- Williams %R(14): Long when crosses above -80 from below (oversold bounce)
  Short when crosses below -20 from above (overbought reversal)
- 1w EMA50 as trend filter: Only long when price > 1w EMA50, short when price < 1w EMA50
- Volume regime: Only trade when current volume > 0.8x 20-period median volume (avoid low-volume false signals)
- ATR trailing stop (2.0x ATR from extreme) for risk management
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
- Williams %R is effective in both bull and bear markets for catching reversals at extremes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume regime filter: > 0.8x 20-period median volume (avoid low-volume noise)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume median, 14 for Williams %R, 50 for 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_median[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams %R crossover signals
        williams_r_prev = williams_r[i-1]
        williams_r_curr = williams_r[i]
        
        # Long signal: Williams %R crosses above -80 from below (oversold bounce)
        long_cross = (williams_r_prev <= -80) and (williams_r_curr > -80)
        # Short signal: Williams %R crosses below -20 from above (overbought reversal)
        short_cross = (williams_r_prev >= -20) and (williams_r_curr < -20)
        
        # Volume regime filter: avoid low-volume false signals
        volume_ok = volume[i] > 0.8 * vol_median[i]
        
        if position == 0:
            # Long: Williams %R bullish crossover + price > 1w EMA50 + volume regime
            if long_cross and close[i] > ema_50_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Williams %R bearish crossover + price < 1w EMA50 + volume regime
            elif short_cross and close[i] < ema_50_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Williams %R crosses below -50 (momentum loss)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            momentum_exit = williams_r[i] < -50
            
            if trailing_stop_long or momentum_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from short extreme (trailing stop)
            # 2. Williams %R crosses above -50 (momentum loss)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            momentum_exit = williams_r[i] > -50
            
            if trailing_stop_short or momentum_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_VolumeRegime_ATRStop"
timeframe = "6h"
leverage = 1.0