#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 Breakout with 1d Supertrend Filter and Volume Spike.
- Camarilla H3/L3 levels from daily chart act as key support/resistance; breakouts capture momentum.
- 1d Supertrend (ATR=10, mult=3.0) provides robust higher-timeframe trend filter to align with intermediate momentum and reduce counter-trend trades.
- Volume spike (>2.0x 24-period average) confirms breakout validity and reduces false signals.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 75-200 total over 4 years (19-50/year) on 4h timeframe to avoid fee drag.
- Works in bull/bear markets via 1d Supertrend filter and volatility-based volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close)):
        if i == 0:
            final_ub.iloc[i] = basic_ub.iloc[i]
            final_lb.iloc[i] = basic_lb.iloc[i]
            supertrend.iloc[i] = final_ub.iloc[i]
            direction.iloc[i] = 1
        else:
            if basic_ub.iloc[i] < final_ub.iloc[i-1] or close.iloc[i-1] > final_ub.iloc[i-1]:
                final_ub.iloc[i] = basic_ub.iloc[i]
            else:
                final_ub.iloc[i] = final_ub.iloc[i-1]
                
            if basic_lb.iloc[i] > final_lb.iloc[i-1] or close.iloc[i-1] < final_lb.iloc[i-1]:
                final_lb.iloc[i] = basic_lb.iloc[i]
            else:
                final_lb.iloc[i] = final_lb.iloc[i-1]
            
            if i == 0:
                supertrend.iloc[i] = final_ub.iloc[i]
                direction.iloc[i] = 1
            else:
                if supertrend.iloc[i-1] == final_ub.iloc[i-1]:
                    if close.iloc[i] <= final_ub.iloc[i]:
                        supertrend.iloc[i] = final_ub.iloc[i]
                    else:
                        supertrend.iloc[i] = final_lb.iloc[i]
                        direction.iloc[i] = -1
                else:
                    if close.iloc[i] >= final_lb.iloc[i]:
                        supertrend.iloc[i] = final_lb.iloc[i]
                    else:
                        supertrend.iloc[i] = final_ub.iloc[i]
                        direction.iloc[i] = 1
    
    return supertrend.values, direction.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Supertrend trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Supertrend trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    supertrend_1d, supertrend_dir_1d = calculate_supertrend(high_1d, low_1d, close_1d, atr_period=10, multiplier=3.0)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    supertrend_dir_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_dir_1d.astype(float))
    
    # Calculate Camarilla pivot levels from 1d OHLC
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla H3 and L3 levels
        camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
        camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
        
        # Align Camarilla levels to 4h timeframe (using previous completed 1d bar)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
    
    # Volume confirmation: > 2.0x 24-period average volume (4h * 6 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_1d_aligned[i]) or np.isnan(supertrend_dir_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 with volume spike and bullish 1d Supertrend (uptrend)
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i] and supertrend_dir_1d_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with volume spike and bearish 1d Supertrend (downtrend)
            elif close[i] < camarilla_l3_aligned[i] and volume_spike[i] and supertrend_dir_1d_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR 1d Supertrend turns bearish
            if close[i] < camarilla_l3_aligned[i] or supertrend_dir_1d_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR 1d Supertrend turns bullish
            if close[i] > camarilla_h3_aligned[i] or supertrend_dir_1d_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dSupertrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0