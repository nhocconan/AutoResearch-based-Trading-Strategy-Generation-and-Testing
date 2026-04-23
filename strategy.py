#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d Volume Regime Filter and ATR Trailing Stop.
- Williams %R(14): Long when < -80 (oversold) and turning up, Short when > -20 (overbought) and turning down
- 1d Volume Regime: Only trade when 1d volume > 1.2x 20-period average (high conviction moves)
- ATR Trailing Stop: 2.0x ATR from extreme for risk management
- Uses 1d volume filter to avoid low-conviction reversals in choppy markets
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
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
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.2x 20-period average (volume filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume and its 20-period MA
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for Williams %R and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams %R reversal conditions
        williams_oversold = williams_r[i] < -80  # Oversold territory
        williams_overbought = williams_r[i] > -20  # Overbought territory
        williams_rising = williams_r[i] > williams_r[i-1]  # Williams %R turning up
        williams_falling = williams_r[i] < williams_r[i-1]  # Williams %R turning down
        
        # 1d volume regime filter: only trade in high volume conviction
        high_volume_regime = volume_1d_aligned[i] > 1.2 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold + turning up + high volume regime
            if williams_oversold and williams_rising and high_volume_regime:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Williams %R overbought + turning down + high volume regime
            elif williams_overbought and williams_falling and high_volume_regime:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Williams %R exits oversold territory (> -50)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            momentum_exit = williams_r[i] > -50
            
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
            # 2. Williams %R exits overbought territory (< -50)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            momentum_exit = williams_r[i] < -50
            
            if trailing_stop_short or momentum_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dVolumeRegime_ATRStop"
timeframe = "6h"
leverage = 1.0