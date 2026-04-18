#!/usr/bin/env python3
"""
4h Triple Supertrend + Volume Spike + 1d Trend Filter
Uses three Supertrend indicators with different ATR multipliers to confirm trend strength.
Long when all three Supertrends are bullish with volume spike.
Short when all three are bearish with volume spike.
Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency with strong trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def supertrend(high, low, close, atr_period, multiplier):
    """Calculate Supertrend indicator."""
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()
    
    hl_avg = (high + low) / 2
    upper = hl_avg + (multiplier * atr)
    lower = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    for i in range(len(close)):
        if i == 0:
            supertrend.iloc[i] = 0.0
            direction.iloc[i] = 1
        else:
            if close.iloc[i] > upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                if direction.iloc[i] < 0 and lower.iloc[i] > lower.iloc[i-1]:
                    lower.iloc[i] = lower.iloc[i-1]
                if direction.iloc[i] > 0 and upper.iloc[i] < upper.iloc[i-1]:
                    upper.iloc[i] = upper.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower.iloc[i]
            else:
                supertrend.iloc[i] = upper.iloc[i]
    
    return supertrend.values, direction.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Triple Supertrend with different ATR multipliers
    st1, dir1 = supertrend(high, low, close, 10, 3.0)  # Standard
    st2, dir2 = supertrend(high, low, close, 10, 3.5)  # Wider
    st3, dir3 = supertrend(high, low, close, 10, 2.5)  # Tighter
    
    # Volume spike detection (2.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(st1[i]) or np.isnan(st2[i]) or np.isnan(st3[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_1d_ema = price > ema_50_1d_aligned[i]
        below_1d_ema = price < ema_50_1d_aligned[i]
        
        # Triple Supertrend alignment
        bullish_alignment = (dir1[i] == 1) and (dir2[i] == 1) and (dir3[i] == 1)
        bearish_alignment = (dir1[i] == -1) and (dir2[i] == -1) and (dir3[i] == -1)
        
        if position == 0:
            # Long: all Supertrends bullish, price above 1d EMA, volume spike
            if (bullish_alignment and above_1d_ema and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: all Supertrends bearish, price below 1d EMA, volume spike
            elif (bearish_alignment and below_1d_ema and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: any Supertrend turns bearish or price breaks below 1d EMA
            if not bullish_alignment or below_1d_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: any Supertrend turns bullish or price breaks above 1d EMA
            if not bearish_alignment or above_1d_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TripleSupertrend_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0